"""Sequence generator -- NO audio dependencies (no numpy).

Consumes a SessionConstitution + VoiceConfig(s) + a shared Arrangement and emits
plain NoteEvent lists. This is the part that will become firmware logic on the
ESP32, so it stays pure: integer/float arithmetic, the portable PRNG, and the
scale tables. It knows nothing about oscillators, filters, or sample rates.

Each voice is still an independent line that auditions well alone, but it now
reads a shared Arrangement (chord progression, phrase clock, energy contour) so
the three lines move with harmonic direction, pulse, and phrasing. On top of the
walk each voice develops short *motifs* (a cell restated and varied), so the ear
gets repetition-with-variation instead of a memoryless random wander.
"""

from dataclasses import dataclass

import scales
from prng import voice_prng
from config import PERC_MIDI, SessionConstitution, VoiceConfig
from arrangement import build_arrangement


@dataclass
class NoteEvent:
    midi: int
    start: float       # seconds
    duration: float    # seconds
    voice: str
    velocity: float = 1.0   # 0..1 loudness; driven by the energy contour
    sound: str = ""    # percussion hit type ("kick"/"snare"/"hat"); "" = pitched note


def _reflect(degree: int, lo: int, hi: int) -> int:
    """Fold a degree back into [lo, hi] by reflecting at the boundaries.

    Unlike clamping, reflection never makes the walk 'stick' at an edge (which
    would repeat the same note); it bounces off. Because every result lands in
    [lo, hi], the walk is provably bounded for any number of steps -- run it
    forever and no note ever escapes the two-octave ambit.
    """
    span = hi - lo
    if span <= 0:
        return lo
    while degree < lo or degree > hi:
        if degree < lo:
            degree = 2 * lo - degree
        else:
            degree = 2 * hi - degree
    return degree


# ---------------------------------------------------------------------------
# Motif development: a short cell of scale-degree deltas the voice restates and
# varies (retrograde / inversion / rotation / augmentation) so a recognizable
# idea keeps returning without ever looping literally. The starting pitch comes
# free from the carried walk position, so each restatement is transposed anew.
# ---------------------------------------------------------------------------
def _new_motif(rng, cfg: VoiceConfig) -> list:
    return [rng.choice(cfg.walk_steps) for _ in range(max(1, cfg.motif_len))]


def _vary_motif(rng, motif: list, cfg: VoiceConfig) -> list:
    r = rng.rand_float()
    if r < 0.30:
        return list(reversed(motif))              # retrograde
    if r < 0.55:
        return [-d for d in motif]                # inversion
    if r < 0.75:
        return motif[1:] + motif[:1]              # rotation
    if r < 0.90:
        m = list(motif)
        i = rng.randint(0, len(m) - 1)
        m[i] = m[i] * 2                           # augment one step
        return m
    return _new_motif(rng, cfg)                   # a fresh idea


def _walk_voice(const: SessionConstitution, cfg: VoiceConfig, voice: str, arr):
    """Arrangement-aware, motif-driven walk over scale degrees for one voice.

    Per bar -> per beat (optionally subdivided), the line advances by its motif's
    next interval, then a note may be emitted with probability shaped by metric
    accent, the energy contour, and call-and-response gating. Notes on strong
    beats bias onto the current chord's tones (forced at cadences), so the voices
    converge harmonically while staying independent lines.
    """
    rng = voice_prng(const.seed, voice)
    scale_len = len(scales.SCALES[const.scale])
    lo, hi = -scale_len, scale_len              # exactly two octaves, centered
    base = scales.root_midi(const.key, 4 + cfg.octave)
    degree = _reflect(cfg.start_degree, lo, hi)
    beats_per_bar = const.beats_per_bar
    slot = const.sec_per_beat
    accents = cfg.beat_accent or (1.0,)

    use_motif = cfg.motif_len > 0
    motif = _new_motif(rng, cfg) if use_motif else []
    motif_pos = 0

    events = []
    for bar in range(const.bars):
        phrase = arr.phrase_index(bar)
        # call-and-response: a voice thins out on phrases of the "wrong" parity
        active = cfg.answer_parity < 0 or (phrase % 2) == cfg.answer_parity
        energy = arr.energy_at(bar)
        chord = arr.chord_tones(bar)
        cadence = arr.is_cadence_bar(bar)
        # develop the motif every motif_bars so the idea evolves
        if use_motif and cfg.motif_bars > 0 and bar > 0 and bar % cfg.motif_bars == 0:
            motif = _vary_motif(rng, motif, cfg)
            motif_pos = 0

        for beat in range(beats_per_bar):
            accent = accents[beat % len(accents)]
            strong = accent >= 1.0
            base_start = (bar * beats_per_bar + beat) * slot
            # occasionally split a beat into eighths for rhythmic surprise; only
            # lightly tied to energy so it still fires in calm passages
            n_sub = 2 if (active and rng.chance(cfg.subdivide * (0.5 + energy))) else 1
            for s in range(n_sub):
                step = motif[motif_pos % len(motif)] if use_motif else rng.choice(cfg.walk_steps)
                motif_pos += 1
                degree = _reflect(degree + step, lo, hi)

                # metric accent x energy x call/response gate the note probability
                dens = cfg.density * accent
                dens *= (1.0 - cfg.energy_response) + cfg.energy_response * (0.5 + energy)
                if not active:
                    dens *= 0.25
                if n_sub == 2:
                    # a subdivided beat is a deliberate eighth-note gesture -- bias
                    # both halves to actually sound so the offbeat lands audibly
                    dens *= 1.7
                if not rng.chance(min(dens, 1.0)):
                    continue

                # chord biasing: strong-beat downbeats land on chord tones; a
                # cadence forces convergence so the voices arrive home together
                pitch_degree = degree
                lock = cfg.chord_lock
                if cadence and strong:
                    lock = max(lock, 0.9)
                if strong and s == 0 and rng.chance(lock):
                    pitch_degree = scales.snap_to_chord(degree, chord, scale_len)

                midi = scales.degree_to_midi(base, const.scale, pitch_degree)
                sub_slot = slot / n_sub
                dur = sub_slot * cfg.note_frac + cfg.decay
                vel = 0.7 + cfg.energy_response * (energy - 0.5) * 0.6
                if strong:
                    vel += 0.08
                vel = max(0.4, min(1.0, vel))
                events.append(NoteEvent(midi, base_start + s * sub_slot, dur, voice, vel))
    return events


def generate_melody(const: SessionConstitution, cfg: VoiceConfig, arr):
    """Lead line: loosely chord-locked, lively rhythm, longest motifs."""
    return _walk_voice(const, cfg, "melody", arr)


def generate_harmony(const: SessionConstitution, cfg: VoiceConfig, arr):
    """Inner voice: tightly chord-locked, calmer, answers the melody's phrases."""
    return _walk_voice(const, cfg, "harmony", arr)


def generate_drone(const: SessionConstitution, cfg: VoiceConfig, arr):
    """Foundation: sustains the current chord's root (sometimes its fifth),
    changing when the progression's chord root changes so it locks the harmony."""
    rng = voice_prng(const.seed, "drone")
    base = scales.root_midi(const.key, 4 + cfg.octave)
    events = []
    bar = 0
    while bar < const.bars:
        root = arr.chord_root(bar)
        # hold until the chord root changes (capped so it still breathes)
        span = 1
        while (bar + span < const.bars
               and arr.chord_root(bar + span) == root
               and span < cfg.change_bars_max):
            span += 1
        # mostly the root; occasionally the chord fifth for openness
        degree = root + 4 if rng.chance(0.25) else root
        midi = scales.degree_to_midi(base, const.scale, degree)
        start = bar * const.sec_per_bar
        dur = span * const.sec_per_bar + cfg.decay
        vel = 0.7 + cfg.energy_response * (arr.energy_at(bar) - 0.5) * 0.6
        vel = max(0.5, min(1.0, vel))
        events.append(NoteEvent(midi, start, dur, "drone", vel))
        bar += span
    return events


# Beatbox groove templates at sixteenth resolution (16 slots per 4/4 bar). Each
# drum has several variants; one is chosen per bar (re-picked with prob
# `variation`) so the pocket shifts without dissolving. Values <1 are ghost
# notes / offbeats that fire probabilistically off the seeded PRNG -- gated by
# `ghost_chance` and energy -- so no two bars land identically. Indexed modulo
# their length, so they still tile if beat_steps is changed.
#                  1              2              3              4
_KICK_VARIANTS = (
    (0.98, 0, 0, 0,     0, 0, 0.15, 0,    0.90, 0, 0, 0.30,   0, 0, 0.20, 0),
    (0.98, 0, 0, 0.30,  0, 0, 0, 0,       0.85, 0, 0.40, 0,   0, 0, 0.25, 0),
    (0.97, 0, 0.20, 0,  0, 0, 0, 0.35,    0.80, 0, 0, 0,      0.30, 0, 0, 0.20),
)
_SNARE_VARIANTS = (
    (0, 0, 0, 0,     0.95, 0, 0, 0.08,    0, 0, 0, 0,      0.95, 0, 0, 0.12),
    (0, 0, 0.10, 0,  0.95, 0, 0, 0,       0, 0.12, 0, 0,   0.95, 0, 0.15, 0),
    (0, 0, 0, 0,     0.95, 0, 0.12, 0,    0, 0, 0.10, 0,   0.95, 0, 0, 0.20),
)
_HAT_VARIANTS = (
    (0.90, 0, 0.80, 0,       0.85, 0, 0.80, 0,       0.90, 0, 0.80, 0,       0.85, 0, 0.80, 0),       # eighths
    (0.90, 0.5, 0.80, 0.5,   0.85, 0.5, 0.80, 0.5,   0.90, 0.5, 0.80, 0.5,   0.85, 0.5, 0.80, 0.6),   # sixteenths
    (0.90, 0, 0.80, 0.55,    0.85, 0, 0.80, 0,       0.90, 0.55, 0.80, 0,    0.85, 0, 0.80, 0.55),    # broken
)


def _perc_dur(cfg: VoiceConfig, sound: str) -> float:
    """Audible length of a hit (its decay plus a small tail)."""
    decays = {
        "kick": cfg.kick_decay, "snare": cfg.snare_decay, "hat": cfg.hat_decay,
        "openhat": cfg.openhat_decay, "clap": cfg.clap_decay,
        "rim": cfg.rim_decay, "tom": cfg.tom_decay,
    }
    return decays.get(sound, cfg.snare_decay) + 0.05


def _make_fill(t0: float, length: float, rng, cfg: VoiceConfig, energy: float) -> list:
    """A random drum fill spanning `length` seconds from `t0`.

    Four flavors (roll / descending toms / stutter / mixed), a random subdivision
    count, and a velocity crescendo into the downbeat -- picked fresh each time so
    fills never repeat. Busier when the energy contour is high.
    """
    kind = rng.randint(0, 3)
    n = rng.choice((6, 8) if energy > 0.55 else (3, 4, 6))
    step = length / n
    evs = []
    for j in range(n):
        t = t0 + j * step
        vel = min(1.0, 0.45 + 0.55 * (j + 1) / n)          # crescendo
        if kind == 0:                                       # snare/rim roll
            snd = "rim" if (j % 2 and rng.chance(0.4)) else "snare"
            midi = PERC_MIDI[snd]
        elif kind == 1:                                     # descending tom run
            snd = "tom"
            midi = PERC_MIDI["tom"] + int(round((1.0 - j / max(n - 1, 1)) * 7))
        elif kind == 2:                                     # snare/tom stutter
            snd = "tom" if j % 2 else "snare"
            midi = PERC_MIDI[snd] + (5 if snd == "tom" else 0)
        else:                                               # kick+snare mix
            snd = rng.choice(("kick", "snare", "snare", "tom"))
            midi = PERC_MIDI[snd]
        evs.append(NoteEvent(midi, t, _perc_dur(cfg, snd), cfg.name, vel, snd))
    return evs


def generate_percussion(const: SessionConstitution, cfg: VoiceConfig, arr):
    """Beatboxer: vocal drum hits (kick/snare/hat/openhat/clap/rim/tom) on the grid.

    Emits no pitched walk. It reads the shared energy contour to get busier on
    swells, morphs between groove variants, sprinkles probabilistic ghost notes,
    open-hat lifts and clap-thickened backbeats, and humanizes every hit's timing
    and velocity -- so the loop never repeats literally. Bars occasionally end in
    a random fill (always into a cadence), and swing pushes offbeats slightly late.
    """
    rng = voice_prng(const.seed, cfg.name)
    steps = max(1, cfg.beat_steps)
    slot = const.sec_per_bar / steps
    steps_per_beat = max(1, steps // const.beats_per_bar)
    hum = cfg.humanize_ms / 1000.0
    events = []

    def emit(midi, base_start, vel, snd):
        """Append a hit with humanized micro-timing and velocity."""
        t = max(0.0, base_start + (rng.rand_float() - 0.5) * 2.0 * hum)
        v = max(0.05, min(1.0, vel * (1.0 + (rng.rand_float() - 0.5) * cfg.humanize_vel)))
        events.append(NoteEvent(midi, t, _perc_dur(cfg, snd), cfg.name, v, snd))

    def fires(pv, energy):
        """Whether a pattern slot fires: primaries almost always, ghosts gated."""
        if pv <= 0.0:
            return False
        p = pv if pv >= 0.85 else pv * cfg.ghost_chance * (0.5 + energy)
        return rng.chance(min(p, 1.0))

    ki = rng.randint(0, len(_KICK_VARIANTS) - 1)
    si = rng.randint(0, len(_SNARE_VARIANTS) - 1)
    hi = rng.randint(0, len(_HAT_VARIANTS) - 1)
    for bar in range(const.bars):
        energy = arr.energy_at(bar)
        cadence = arr.is_cadence_bar(bar)
        # morph the groove: occasionally swap each drum's variant
        if rng.chance(cfg.variation):
            ki = rng.randint(0, len(_KICK_VARIANTS) - 1)
        if rng.chance(cfg.variation):
            si = rng.randint(0, len(_SNARE_VARIANTS) - 1)
        if rng.chance(cfg.variation):
            hi = rng.randint(0, len(_HAT_VARIANTS) - 1)
        kick_p, snare_p, hat_p = _KICK_VARIANTS[ki], _SNARE_VARIANTS[si], _HAT_VARIANTS[hi]

        # decide a fill and how much of the bar's tail it takes over
        fill = (cadence and rng.chance(0.7)) or rng.chance(cfg.fill_chance * (0.3 + energy))
        big = fill and (cadence or energy > 0.6) and rng.chance(0.4)
        fill_from = steps - steps_per_beat * (2 if big else 1) if fill else steps

        for i in range(steps):
            if i >= fill_from:
                break
            downbeat = (i % steps_per_beat) == 0
            base_start = bar * const.sec_per_bar + i * slot
            if i % 2 == 1:                      # swing the offbeats a touch late
                base_start += cfg.swing * slot

            if fires(kick_p[i % len(kick_p)], energy):
                emit(PERC_MIDI["kick"], base_start, 0.95 if downbeat else 0.62, "kick")

            sv = snare_p[i % len(snare_p)]
            if fires(sv, energy):
                if sv >= 0.85:                 # backbeat: sometimes thickened by a clap
                    emit(PERC_MIDI["snare"], base_start, 0.85, "snare")
                    if rng.chance(0.2 + 0.35 * energy):
                        emit(PERC_MIDI["clap"], base_start, 0.55, "clap")
                else:                          # ghost: a quiet rim or snare tap
                    snd = "rim" if rng.chance(0.5) else "snare"
                    emit(PERC_MIDI[snd], base_start, 0.4, snd)

            if fires(hat_p[i % len(hat_p)], energy):
                last_of_beat = (i % steps_per_beat) == steps_per_beat - 1
                if last_of_beat and rng.chance(0.15 + 0.25 * energy):
                    emit(PERC_MIDI["openhat"], base_start, 0.7, "openhat")   # open-hat lift
                else:
                    emit(PERC_MIDI["hat"], base_start, 0.5 if downbeat else 0.7, "hat")

        if fill:
            t0 = bar * const.sec_per_bar + fill_from * slot
            events.extend(_make_fill(t0, (steps - fill_from) * slot, rng, cfg, energy))
            # crash-style cap on the next downbeat
            if bar + 1 < const.bars and rng.chance(0.6):
                snd = "openhat" if rng.chance(0.6) else "clap"
                emit(PERC_MIDI[snd], (bar + 1) * const.sec_per_bar, 0.9, snd)
    return events


_GENERATORS = {
    "melody": generate_melody,
    "harmony": generate_harmony,
    "drone": generate_drone,
    "beat": generate_percussion,
}


def generate_voice(const: SessionConstitution, cfg: VoiceConfig, arr=None):
    """Generate events for a single named voice, building an arrangement if none."""
    if arr is None:
        arr = build_arrangement(const)
    return _GENERATORS[cfg.name](const, cfg, arr)


def generate_session(const: SessionConstitution, voices: dict, arr=None):
    """Generate every voice over one shared arrangement; {voice_name: [events]}."""
    if arr is None:
        arr = build_arrangement(const)
    return {name: generate_voice(const, cfg, arr) for name, cfg in voices.items()}
