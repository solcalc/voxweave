"""Session + voice configuration and the vowel formant table.

These dataclasses are the single place to tweak a render. Nothing in the core
generator or synth hardcodes musical constants -- they read from here. The shapes
(flat tuples, small dicts) mirror what would become C structs / lookup tables.
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Vowel formant table.
# Each vowel maps to three formants (F1, F2, F3), each a (freq_hz, Q, gain).
# These are the parallel bandpass centers that shape a sawtooth into a "voice".
# Same flat shape as a C table:  static const Formant OO[3] = {{...},{...},{...}};
# ---------------------------------------------------------------------------
VOWEL_FORMANTS = {
    # freq (Hz),   Q,    gain
    "ee": [
        (270.0, 6.0, 1.00),
        (2290.0, 10.0, 0.50),
        (3010.0, 12.0, 0.30),
    ],
    "oh": [
        (570.0, 6.0, 1.00),
        (840.0, 8.0, 0.55),
        (2410.0, 12.0, 0.20),
    ],
    "oo": [
        (300.0, 6.0, 1.00),
        (870.0, 8.0, 0.40),
        (2240.0, 12.0, 0.15),
    ],
}


# ---------------------------------------------------------------------------
# Percussion "mouth" colors -- the beatbox voice.
# Same (freq, Q, gain) formant shape as the vowels, but tuned to color *noise*
# (and a kick's body) into vocal-sounding drum hits rather than sung vowels, so
# the beat still reads as coming from a mouth. Kept out of VOWEL_FORMANTS so the
# sung voices' vowel menu stays clean; the percussion synth reads this directly.
# Ports to C as another static const Formant[3] table.
# ---------------------------------------------------------------------------
PERC_FORMANTS = {
    # freq (Hz),   Q,    gain
    "kick": [                       # closed-lips "b" boom: low, rounded
        (150.0, 2.0, 1.00),
        (450.0, 3.0, 0.35),
        (900.0, 5.0, 0.12),
    ],
    "snare": [                      # "psh"/"eh" burst: bright mid body
        (500.0, 2.5, 0.70),
        (1650.0, 3.0, 1.00),
        (2900.0, 4.0, 0.65),
    ],
    "hat": [                        # "ts" fricative: airy top end
        (5200.0, 3.0, 0.70),
        (8200.0, 4.0, 1.00),
        (11500.0, 5.0, 0.55),
    ],
    "openhat": [                    # "tss" -- an open hat: a touch lower/looser
        (4600.0, 2.5, 0.70),
        (7400.0, 3.0, 1.00),
        (10500.0, 4.0, 0.60),
    ],
    "clap": [                       # "pf" hand-clap: noisy mid band
        (1000.0, 2.0, 0.80),
        (1750.0, 2.5, 1.00),
        (2700.0, 3.0, 0.60),
    ],
    "rim": [                        # "tk" rim/side-stick: short, bright click
        (1800.0, 3.0, 0.60),
        (2700.0, 3.5, 1.00),
        (3600.0, 4.0, 0.70),
    ],
    "tom": [                        # "dm" tom: low-mid rounded body
        (250.0, 2.0, 1.00),
        (520.0, 3.0, 0.40),
        (950.0, 4.0, 0.12),
    ],
}

# Nominal MIDI notes per hit so they still plot on the piano roll and so the
# generator and synth agree on pitch (GM drum map). Shared by both -- the tom
# fills encode a descending pitch as an offset from PERC_MIDI["tom"].
PERC_MIDI = {
    "kick": 36, "snare": 38, "hat": 42, "openhat": 46,
    "clap": 39, "rim": 37, "tom": 45,
}


@dataclass
class SessionConstitution:
    """Shared 'constitution' all three voices derive from, so they stay in tune.

    Broadcast equivalent: (key, scale, tempo, seed, bars) is what a lead doll
    would send over ESP-NOW; every doll reconstructs the same session from it.
    """

    key: str = "C"
    scale: str = "dorian"
    tempo: int = 60           # BPM (ambient default: slow, 40-90)
    seed: int = 42
    bars: int = 16
    beats_per_bar: int = 4
    sample_rate: int = 44100
    # --- arrangement (shared harmonic structure all voices read) ---
    phrase_bars: int = 4      # bars per phrase; cadence lands on the last bar
    chord_bars: int = 2       # harmonic rhythm: hold each chord this many bars

    @property
    def sec_per_beat(self) -> float:
        return 60.0 / self.tempo

    @property
    def sec_per_bar(self) -> float:
        return self.sec_per_beat * self.beats_per_bar

    @property
    def total_seconds(self) -> float:
        return self.sec_per_bar * self.bars


@dataclass
class VoiceConfig:
    """Per-voice knobs. `octave` is the register offset in octaves from C4."""

    name: str
    vowel: str
    octave: int              # register offset (+ = higher)
    density: float           # 0..1 probability a slot plays vs rests
    # allowed random-walk steps in scale degrees
    walk_steps: tuple = (-2, -1, 1, 2)
    # where the walk begins, in scale degrees relative to the base register
    # (0 = base root). Just biases the starting pitch; the walk wanders from there.
    start_degree: int = 0
    # drone-only: pool of scale degrees to sustain, and how often it changes
    degree_pool: tuple = (0, 4)
    change_bars_min: int = 8
    change_bars_max: int = 16
    # envelope (seconds). Ambient: fast-ish attack, long exponential decay.
    attack: float = 0.02
    decay: float = 2.5
    # note length as a fraction of a beat's grid slot (before decay tail)
    note_frac: float = 0.9
    amp: float = 0.9

    # --- arrangement response (how this voice reads the shared structure) ---
    # Probability a note on a strong beat snaps to the current chord's tones.
    # Weak beats/subdivisions stay as walked (passing tones). Cadence bars force
    # this higher so voices converge. Drone follows the root directly (see below).
    chord_lock: float = 0.6
    # Per-beat density multipliers (metric accent): index by beat-in-bar, wrapping.
    # Strong beats (1 and 3) favored so a pulse emerges from the flat grid.
    beat_accent: tuple = (1.3, 0.7, 1.1, 0.6)
    # Probability a beat splits into two eighth notes (scaled by energy).
    subdivide: float = 0.25
    # Motif development: a short cell of scale-degree deltas the voice restates
    # and varies for `motif_bars` before regenerating -- repetition-with-variation.
    motif_len: int = 4
    motif_bars: int = 2
    # How strongly density/amplitude/register track the 0..1 energy contour.
    energy_response: float = 0.6
    # Call-and-response: the voice thins out on phrases whose parity != this.
    # 0 or 1 to trade phrases with another voice; -1 to always play (drone).
    answer_parity: int = -1

    # --- "voice-likeness" controls (all cheap / fixed-point portable) ---
    # Vibrato: a sine LFO on pitch, eased in after a delay (singers don't start
    # a note with vibrato -- it blooms a moment in). depth is in semitones.
    vibrato_hz: float = 5.5
    vibrato_semitones: float = 0.35
    vibrato_delay: float = 0.35      # seconds before vibrato starts
    vibrato_ramp: float = 0.6        # seconds to ease vibrato to full depth
    # Tremolo: amplitude wobble coupled to the vibrato LFO (0..1 of full swing).
    tremolo: float = 0.12
    # Pitch scoop: glide up into the target pitch at the note onset.
    scoop_semitones: float = 0.4
    scoop_time: float = 0.07         # time-constant of the scoop (seconds)
    # Jitter/shimmer: tiny per-note random detune (cents) and amplitude (fraction).
    jitter_cents: float = 12.0
    shimmer: float = 0.06
    # Aspiration/breath: vowel-colored noise mixed into the source (0..1).
    breath: float = 0.06
    # Glottal spectral tilt: one-pole lowpass cutoff (Hz) to tame sawtooth buzz.
    tilt_hz: float = 2800.0

    # --- percussion / beatbox voice (only used when percussion=True) ---
    # When set, this voice ignores pitch/vowel/vibrato entirely and is rendered
    # as vocal-sounding drum hits (kick/snare/hat) laid on the beat grid.
    percussion: bool = False
    beat_steps: int = 16             # rhythmic slots per bar (16 = sixteenth notes)
    swing: float = 0.12              # push offbeat slots later, as a fraction of a slot
    # --- groove variation & humanization (what keeps it from sounding looped) ---
    variation: float = 0.3           # per-bar chance to re-pick each drum's pattern variant
    ghost_chance: float = 0.6        # scales the ghost-note / offbeat probabilities (0..1+)
    humanize_ms: float = 6.0         # random micro-timing jitter per hit (ms)
    humanize_vel: float = 0.18       # random per-hit velocity variation (fraction)
    # --- drum fills: an occasional bar ends with a random roll/tom run ---
    fill_chance: float = 0.18        # base chance a bar ends in a fill (scaled by energy;
                                     # forced high on cadence bars)
    # kick ("b"): a short pitched thump that drops in pitch, plus a lip-release click.
    click: float = 0.5               # amount of the noisy lip-release transient
    kick_hz: float = 105.0           # kick body start pitch
    kick_hz_end: float = 45.0        # pitch it falls to
    kick_pitch_tau: float = 0.03     # kick pitch-drop time constant (s)
    kick_decay: float = 0.16         # kick amplitude decay (s)
    # snare ("psh"): colored noise over a short tonal body.
    snare_tone_hz: float = 190.0     # tonal body under the snare noise
    snare_noise: float = 0.85        # noise vs tone balance (1 = all noise)
    snare_decay: float = 0.11        # snare amplitude decay (s)
    # hat ("ts") / open hat ("tss") / clap ("pf") / rim ("tk") / tom ("dm").
    hat_decay: float = 0.045         # closed-hat amplitude decay (s)
    openhat_decay: float = 0.22      # open-hat amplitude decay (s)
    clap_decay: float = 0.09         # clap tail decay (s)
    rim_decay: float = 0.03          # rim/side-stick decay (s)
    tom_hz: float = 150.0            # tom body pitch (fills detune this per hit)
    tom_decay: float = 0.20          # tom amplitude decay (s)


def default_voices() -> dict:
    """The three established dolls: melody=ee upper, harmony=oh mid, drone=oo low."""
    return {
        "melody": VoiceConfig(
            name="melody", vowel="ee", octave=1, density=0.55,
            walk_steps=(-2, -1, 1, 2), attack=0.06, decay=1.8, note_frac=0.8,
            vibrato_hz=5.7, vibrato_semitones=0.4, scoop_semitones=0.5,
            # lead line: loosely chord-locked (room to roam), lively subdivisions,
            # develops the longest motifs, and answers on odd phrases
            chord_lock=0.5, subdivide=0.4, motif_len=4, motif_bars=2,
            energy_response=0.7, answer_parity=1,
        ),
        "harmony": VoiceConfig(
            name="harmony", vowel="oh", octave=0, density=0.35, start_degree=2,
            attack=0.08, decay=2.5, note_frac=0.9,
            vibrato_hz=5.2, vibrato_semitones=0.3,
            # inner voice: tightly chord-locked, calmer rhythm, answers on even
            # phrases so it fills the melody's rests (call-and-response)
            chord_lock=0.75, subdivide=0.1, motif_len=3, motif_bars=2,
            energy_response=0.5, answer_parity=0,
        ),
        "drone": VoiceConfig(
            name="drone", vowel="oo", octave=-1, density=1.0,
            degree_pool=(0, 3, 4), change_bars_min=8, change_bars_max=16,
            attack=0.4, decay=6.0, note_frac=1.0, amp=0.7,
            # a drone breathes slowly: gentle, slow vibrato, no scoop, more air
            vibrato_hz=4.6, vibrato_semitones=0.18, vibrato_delay=0.8,
            vibrato_ramp=1.5, scoop_semitones=0.0, tremolo=0.08, breath=0.09,
            # foundation: tracks the chord root directly, never subdivides or
            # rests, and always sounds (answer_parity=-1)
            chord_lock=1.0, subdivide=0.0, energy_response=0.25, answer_parity=-1,
        ),
        "beat": VoiceConfig(
            name="beat", vowel="oo", octave=0, density=1.0,
            # a beatboxer: unpitched vocal drum hits on the grid. vowel/octave are
            # inert here (percussion path). decay=0 keeps the piano-roll blocks tight.
            percussion=True, amp=0.85, decay=0.0,
            # tracks the energy contour (busier hats/ghost notes on swells) and,
            # like the drone, always plays (answer_parity=-1)
            energy_response=0.55, answer_parity=-1,
        ),
    }
