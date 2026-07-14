"""DSP / synthesis -- numpy only. No scipy, no librosa.

The signal chain per note is:
    sawtooth oscillator -> 3 parallel bandpass biquads (formant) -> ADSR -> buffer

The biquad coefficient math is written out by hand (RBJ audio-EQ cookbook) so it
ports almost line-for-line to fixed-point C on the ESP32. Everything works at a
fixed sample rate and avoids anything that needs arbitrary-precision floats.
"""

import math

import numpy as np

from config import PERC_FORMANTS, PERC_MIDI, VOWEL_FORMANTS, SessionConstitution, VoiceConfig
from scales import midi_to_hz

# The two recursive filters below (one-pole and biquad) are the whole render
# cost: each is a per-sample feedback loop that can't be vectorized in numpy.
# We JIT them with numba when it's available -- the loop body is left exactly as
# it would be in C, so it still ports line-for-line to the ESP32. Without numba
# the same functions run as plain Python (correct, just slow); the desktop
# testbed installs numba (see requirements.txt) to make renders near-instant.
try:
    from numba import njit
except ImportError:  # pragma: no cover - fallback keeps pure-numpy behaviour
    def njit(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap(args[0]) if args and callable(args[0]) else wrap


@njit(cache=True)
def _one_pole_lp_kernel(x, a, b):
    y = np.empty_like(x)
    yp = 0.0
    for n in range(x.shape[0]):
        yp = b * x[n] + a * yp
        y[n] = yp
    return y


@njit(cache=True)
def _biquad_kernel(x, b0, b1, b2, a1, a2):
    y = np.empty_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for n in range(x.shape[0]):
        xn = x[n]
        yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[n] = yn
        x2, x1 = x1, xn
        y2, y1 = y1, yn
    return y


# ---------------------------------------------------------------------------
# Oscillator
# ---------------------------------------------------------------------------
def sawtooth(freq: float, n: int, sr: int) -> np.ndarray:
    """Naive phase-accumulator sawtooth in [-1, 1].

    Uses an explicit per-sample phase increment (like a fixed-point phase
    accumulator would on the MCU), not a closed-form vectorized trick, so the
    structure matches the C port.
    """
    inc = freq / sr                       # cycles per sample
    phase = (np.arange(n, dtype=np.float64) * inc) % 1.0
    return 2.0 * phase - 1.0              # ramp 0..1 -> -1..1


def sawtooth_fm(freq_inst: np.ndarray, sr: int) -> np.ndarray:
    """Sawtooth whose frequency varies per sample (for vibrato/scoop/jitter).

    `freq_inst[n]` is the instantaneous frequency in Hz. Phase is the running
    sum of per-sample increments -- exactly a fixed-point phase accumulator on
    the MCU, where the increment register is updated each sample from the LFO.
    """
    inc = freq_inst / sr
    phase = np.cumsum(inc) % 1.0
    return 2.0 * phase - 1.0


def one_pole_lp(x: np.ndarray, cutoff: float, sr: int) -> np.ndarray:
    """One-pole lowpass:  y[n] = b*x[n] + a*y[n-1].

    Approximates the glottal source's ~-12 dB/oct rolloff so the raw sawtooth
    stops sounding buzzy/reedy. Trivial to port (one multiply-add per sample).
    """
    a = math.exp(-2.0 * math.pi * cutoff / sr)
    b = 1.0 - a
    return _one_pole_lp_kernel(x, a, b)


# ---------------------------------------------------------------------------
# Biquad bandpass -- RBJ cookbook, coefficients explicit
# ---------------------------------------------------------------------------
class Biquad:
    """Single bandpass biquad (constant 0 dB peak gain variant of RBJ BPF).

    Direct Form I. The process() loop is intentionally a plain sample loop so it
    reads like the C you'd write:  y = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2;
    """

    def __init__(self, f0: float, q: float, sr: int):
        w0 = 2.0 * math.pi * f0 / sr
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * q)

        # Bandpass, constant 0 dB peak gain (RBJ):
        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        # Normalize by a0 so a0 == 1 (as we'd bake in for the MCU).
        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = a1 / a0
        self.a2 = a2 / a0

    def process(self, x: np.ndarray) -> np.ndarray:
        return _biquad_kernel(x, self.b0, self.b1, self.b2, self.a1, self.a2)


def formant_filter(x: np.ndarray, vowel, sr: int) -> np.ndarray:
    """Three parallel bandpass biquads at F1/F2/F3, gain-weighted and summed.

    `vowel` is either a name in VOWEL_FORMANTS or an explicit list of
    (freq, Q, gain) formants (used by the percussion voice's mouth colors).
    """
    formants = VOWEL_FORMANTS[vowel] if isinstance(vowel, str) else vowel
    out = np.zeros_like(x)
    for freq, q, gain in formants:
        bq = Biquad(freq, q, sr)
        out += gain * bq.process(x)
    return out


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------
def adsr_envelope(n: int, sr: int, attack: float, decay: float) -> np.ndarray:
    """Fast-ish linear attack then exponential decay over the note's length.

    Simplified ADSR: attack ramp to 1.0, then exp decay with time-constant `decay`.
    """
    env = np.empty(n, dtype=np.float64)
    a = min(int(attack * sr), n)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a, endpoint=False)
    # exponential decay for the remainder
    d = n - a
    if d > 0:
        t = np.arange(d, dtype=np.float64) / sr
        env[a:] = np.exp(-t / max(decay, 1e-4))
    return env


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _total_samples(events, const: SessionConstitution) -> int:
    end = const.total_seconds
    for ev in events:
        end = max(end, ev.start + ev.duration)
    return int(math.ceil(end * const.sample_rate)) + 1


def _pitch_and_amp_mod(n: int, sr: int, cfg: VoiceConfig, rng):
    """Per-sample pitch offset (semitones) and amplitude modulation for a note.

    Combines vibrato (delayed, eased-in sine LFO), a pitch scoop into the note,
    a constant per-note jitter detune, and a coupled tremolo. All deterministic
    given the seeded per-note `rng`, so renders stay reproducible.
    """
    t = np.arange(n, dtype=np.float64) / sr
    # vibrato eases in: 0 before `delay`, ramping to 1 over `ramp` seconds
    onset = np.clip((t - cfg.vibrato_delay) / max(cfg.vibrato_ramp, 1e-4), 0.0, 1.0)
    lfo = np.sin(2.0 * math.pi * cfg.vibrato_hz * t)
    vibrato = cfg.vibrato_semitones * onset * lfo
    # scoop: start flat below the pitch and glide up (decaying offset)
    scoop = -cfg.scoop_semitones * np.exp(-t / max(cfg.scoop_time, 1e-4))
    # jitter: one small random detune per note (a few cents), constant over it
    jitter = (cfg.jitter_cents / 100.0) * rng.standard_normal()
    semis = vibrato + scoop + jitter
    # tremolo rides the same LFO (amplitude dips/rises with the vibrato)
    tremolo = 1.0 + cfg.tremolo * onset * lfo
    return semis, tremolo


# ---------------------------------------------------------------------------
# Percussion (beatbox) synthesis -- vocal-sounding drum hits.
# Each hit is a short noise/tone burst shaped by a "mouth" formant color
# (PERC_FORMANTS) so kick/snare/hat read as coming from a mouth, not a drum
# machine. Same cheap primitives as the sung voices (sine, noise, biquads,
# exp env), so it ports to the ESP32 the same way.
# ---------------------------------------------------------------------------
def _perc_kick(n: int, sr: int, cfg: VoiceConfig, rng, ev) -> np.ndarray:
    """A "b": short pitched thump dropping in pitch, plus a lip-release click."""
    t = np.arange(n, dtype=np.float64) / sr
    f = cfg.kick_hz_end + (cfg.kick_hz - cfg.kick_hz_end) * np.exp(-t / max(cfg.kick_pitch_tau, 1e-4))
    body = np.sin(2.0 * math.pi * np.cumsum(f) / sr)
    # the plosive lip-release: a very short colored-noise transient at the front
    click = formant_filter(rng.standard_normal(n), PERC_FORMANTS["kick"], sr)
    click *= np.exp(-t / 0.008)
    env = adsr_envelope(n, sr, 0.002, cfg.kick_decay)
    return (body + cfg.click * click) * env


def _perc_snare(n: int, sr: int, cfg: VoiceConfig, rng, ev) -> np.ndarray:
    """A "psh": colored breath noise over a brief tonal body."""
    t = np.arange(n, dtype=np.float64) / sr
    body = formant_filter(rng.standard_normal(n), PERC_FORMANTS["snare"], sr)
    tone = np.sin(2.0 * math.pi * cfg.snare_tone_hz * t)
    src = cfg.snare_noise * body + (1.0 - cfg.snare_noise) * tone
    env = adsr_envelope(n, sr, 0.002, cfg.snare_decay)
    return src * env


def _perc_hat(n: int, sr: int, cfg: VoiceConfig, rng, ev) -> np.ndarray:
    """A "ts": a very short airy fricative burst."""
    body = formant_filter(rng.standard_normal(n), PERC_FORMANTS["hat"], sr)
    env = adsr_envelope(n, sr, 0.001, cfg.hat_decay)
    return body * env


def _perc_openhat(n: int, sr: int, cfg: VoiceConfig, rng, ev) -> np.ndarray:
    """A "tss": like the hat but left open, so it rings longer."""
    body = formant_filter(rng.standard_normal(n), PERC_FORMANTS["openhat"], sr)
    env = adsr_envelope(n, sr, 0.001, cfg.openhat_decay)
    return body * env


def _perc_clap(n: int, sr: int, cfg: VoiceConfig, rng, ev) -> np.ndarray:
    """A "pf" hand-clap: a few fast noise transients then a short tail."""
    t = np.arange(n, dtype=np.float64) / sr
    body = formant_filter(rng.standard_normal(n), PERC_FORMANTS["clap"], sr)
    env = np.zeros(n, dtype=np.float64)
    for d in (0.0, 0.008, 0.016):        # three quick claps
        env += np.exp(-(t - d) / 0.006) * (t >= d)
    env += 0.6 * np.exp(-t / max(cfg.clap_decay, 1e-4))   # settling tail
    peak = env.max()
    if peak > 0.0:
        env /= peak
    return body * env


def _perc_rim(n: int, sr: int, cfg: VoiceConfig, rng, ev) -> np.ndarray:
    """A "tk" rim/side-stick: a short, bright click."""
    body = formant_filter(rng.standard_normal(n), PERC_FORMANTS["rim"], sr)
    env = adsr_envelope(n, sr, 0.0005, cfg.rim_decay)
    return body * env


def _perc_tom(n: int, sr: int, cfg: VoiceConfig, rng, ev) -> np.ndarray:
    """A "dm" tom: a pitched body that drops a little; fills detune it per hit."""
    t = np.arange(n, dtype=np.float64) / sr
    # pitch encoded as a semitone offset from PERC_MIDI["tom"] (descending fills)
    base = cfg.tom_hz * 2.0 ** ((ev.midi - PERC_MIDI["tom"]) / 12.0)
    f = 0.7 * base + 0.3 * base * np.exp(-t / 0.05)
    body = np.sin(2.0 * math.pi * np.cumsum(f) / sr)
    click = formant_filter(rng.standard_normal(n), PERC_FORMANTS["tom"], sr)
    click *= np.exp(-t / 0.006)
    env = adsr_envelope(n, sr, 0.002, cfg.tom_decay)
    # the click is a broadband attack transient; the pads mask the sustained sine
    # body but not this brief front, so leaning on it lets the roll punch through.
    return (body + 0.8 * click) * env * cfg.tom_level


_PERC_RENDER = {
    "kick": _perc_kick, "snare": _perc_snare, "hat": _perc_hat,
    "openhat": _perc_openhat, "clap": _perc_clap, "rim": _perc_rim, "tom": _perc_tom,
}

# Per-hit output gains so the sounds sit in balance (noise-through-formant comes
# out quieter than a raw sine body); tuned by ear on the testbed.
_PERC_GAIN = {
    "kick": 1.0, "snare": 3.2, "hat": 2.6, "openhat": 2.4,
    "clap": 2.8, "rim": 3.0, "tom": 1.3,
}


def render_percussion(events, const: SessionConstitution, cfg: VoiceConfig) -> np.ndarray:
    """Render the beatbox voice's NoteEvents (each tagged with a `sound`)."""
    sr = const.sample_rate
    buf = np.zeros(_total_samples(events, const), dtype=np.float64)
    for ev in events:
        n = int(ev.duration * sr)
        if n <= 0:
            continue
        sound = ev.sound or "kick"
        render = _PERC_RENDER.get(sound)
        if render is None:
            continue
        # deterministic per-hit RNG (seed from sound + onset) so the colored
        # noise is reproducible, matching the sung voices' seeding discipline
        rng = np.random.default_rng((hash(sound) & 0xFFFF) ^ int(ev.start * 1000.0))
        velocity = getattr(ev, "velocity", 1.0)
        hit = cfg.amp * velocity * _PERC_GAIN.get(sound, 1.0) * render(n, sr, cfg, rng, ev)
        start = int(ev.start * sr)
        buf[start:start + n] += hit
    # soft-limit: noise bursts have a high crest factor, so raw transient peaks
    # run past 1.0 while the body sits near ~0.1. tanh is ~linear there (loudness
    # and the kick/snare/hat balance are preserved) but folds the tips back under
    # 1.0 so solo renders don't clip and the beat doesn't hog mix normalization --
    # and the gentle saturation suits a mouth-made beat.
    return np.tanh(buf)


def render_events(events, const: SessionConstitution, cfg: VoiceConfig) -> np.ndarray:
    """Render one voice's NoteEvents to a float buffer."""
    if getattr(cfg, "percussion", False):
        return render_percussion(events, const, cfg)
    sr = const.sample_rate
    buf = np.zeros(_total_samples(events, const), dtype=np.float64)
    for ev in events:
        n = int(ev.duration * sr)
        if n <= 0:
            continue
        # per-note deterministic RNG (seed from pitch+onset) for jitter/breath
        rng = np.random.default_rng((ev.midi << 20) ^ int(ev.start * 1000.0))
        freq = midi_to_hz(ev.midi)
        env = adsr_envelope(n, sr, cfg.attack, cfg.decay)

        # pitch modulation -> instantaneous frequency -> FM sawtooth source
        semis, tremolo = _pitch_and_amp_mod(n, sr, cfg, rng)
        freq_inst = freq * np.exp2(semis / 12.0)
        source = sawtooth_fm(freq_inst, sr)
        # glottal spectral tilt: tame the buzz before formant shaping
        if cfg.tilt_hz > 0.0:
            source = one_pole_lp(source, cfg.tilt_hz, sr)
        # aspiration: vowel-colored breath noise, loudest at the onset
        if cfg.breath > 0.0:
            source = source + cfg.breath * rng.standard_normal(n)

        shaped = formant_filter(source, cfg.vowel, sr)
        # per-note amplitude shimmer + coupled tremolo, scaled by the note's
        # velocity so the arrangement's energy contour is audible as swells
        shimmer = 1.0 + cfg.shimmer * rng.standard_normal()
        velocity = getattr(ev, "velocity", 1.0)
        note = cfg.amp * velocity * shimmer * shaped * env * tremolo
        start = int(ev.start * sr)
        buf[start:start + n] += note
    return buf


def mix(buffers) -> np.ndarray:
    """Sum equal-length-or-not buffers and normalize to avoid clipping."""
    length = max((b.shape[0] for b in buffers), default=0)
    out = np.zeros(length, dtype=np.float64)
    for b in buffers:
        out[:b.shape[0]] += b
    peak = np.max(np.abs(out)) if out.size else 0.0
    if peak > 1.0:
        out /= peak
    return out
