# voxweave

A Python engine for generating harmonized, seed-reproducible ambient music from
four synchronized vocal-formant voices. Built as a testbed for an algorithm that
will eventually run on ESP32-WROOM-32U hardware (ESP-IDF + MAX98357A I2S) —
iterate here, listen, then port to C.

## Concept

Four synchronized "doll" voices derive from one shared **constitution**
(root key, scale, tempo, seed) so they stay harmonized:

- **melody** — random walk across scale degrees, sparse rhythm, upper register, vowel `ee`
- **harmony** — its own independent random walk on the same key/scale, mid register, sparser, vowel `oh`
- **drone** — sustains one scale degree, changes every 8–16 bars, low register, vowel `oo`
- **beat** — a beatboxer: vocal-sounding kick/snare/hat/open-hat/clap/rim/tom hits on a sixteenth grid, shaped by "mouth" formants. Morphs between groove variants, sprinkles humanized ghost notes, and drops random fills (and always fills into a cadence), so it never loops literally

Each voice is a self-contained musical line (auditions well alone) yet stays in
key with the others because they share the constitution. Every walking voice is
bounded to **two octaves** (±1 octave around its base register) by reflecting off
the boundaries — provably bounded no matter how long it runs, without the
note-repetition artifact that hard clamping would cause.

Synthesis per note: sawtooth → 3 parallel bandpass biquads (formant filter, RBJ
cookbook) → ADSR envelope → mix.

## Module layout (portability boundaries are deliberate)

| file           | deps            | role |
|----------------|-----------------|------|
| `prng.py`      | none            | portable xorshift32 PRNG (matches on-device C) |
| `scales.py`    | none            | scale semitone tables, MIDI/Hz helpers |
| `config.py`    | none            | `SessionConstitution`, `VoiceConfig`, vowel formant table |
| `generator.py` | prng/scales/config | note-event generation — **no audio deps** |
| `synth.py`     | numpy           | oscillator, biquad, ADSR, formant, mix — **numpy only** |
| `wav_io.py`    | stdlib wave / sounddevice | WAV write + optional playback |
| `testbed.py`   | all above       | argparse CLI |

`generator.py` never imports numpy; `synth.py` never imports scipy/librosa.

## Setup (Arch: Python is externally-managed, use a venv)

```bash
cd python-testbed
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt      # numpy required; sounddevice optional
# for --play only:  sudo pacman -S portaudio
```

## Usage

```bash
# one voice in isolation
python testbed.py --voice melody --key C --scale dorian --seed 42
python testbed.py --voice harmony --seed 42
python testbed.py --voice drone --seed 42
python testbed.py --voice beat --seed 42

# all four, plus the mix
python testbed.py --voice all --seed 42

# vary parameters (all tunable via CLI, nothing hardcoded)
python testbed.py --voice all --scale lydian --key G --tempo 50 --bars 24
python testbed.py --voice all --seed 7 --density-melody 0.7 --density-harmony 0.5

# listen directly (falls back to WAV-only if sounddevice/portaudio absent)
python testbed.py --voice all --seed 42 --play
```

Renders land in `output/` with descriptive names, e.g.
`output/seed42_C-dorian_all.wav`, so you can A/B compare past runs.

## GUI

A Dear PyGui desktop front-end wraps the same pipeline for interactive work:

```bash
python gui.py
```

Edit every session and per-voice parameter live, toggle each voice on/off, and
**Generate** to render — each voice is drawn as a read-only piano roll (pitch vs
time, one colour per voice). **Play**/**Stop** audition the mix (via sounddevice),
**Export WAV** writes to `output/`. It calls `generate_voice` / `render_events` /
`mix` directly, so an untouched render is byte-identical to the equivalent
`testbed.py` run.

## Reproducibility

Generation uses an explicit xorshift32 seeded from `--seed` (never numpy's global
random state), so the same args always produce byte-identical WAVs — the same
property the real dolls need when reconstructing a session from a seed broadcast
over ESP-NOW.
