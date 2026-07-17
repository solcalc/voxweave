# voxweave on ESP32-WROOM — a "how it works under the hood" walkthrough

This is a **teaching port**, not the real firmware. It's a stripped-down slice of
`synth.py` rewritten in C the way it would run on the ESP32-WROOM, and it builds
and runs **on your desktop** so you can read the code and hear the result at the
same time. Only `stream.c`'s *output* is faked (it writes a WAV instead of
driving an I2S DAC); the synthesis is exactly what you'd flash.

## Run it

```
make run        # builds, renders out.wav (~9s), prints what happened
aplay out.wav   # or open out.wav in any player
make clean
```

## Read it in this order

| File | What to learn from it | Python twin |
|------|----------------------|-------------|
| **`dsp.h`** | The 4 synthesis primitives as one-sample-at-a-time **state machines**. This is *the* change that makes streaming possible. | `synth.py` kernels (`_biquad_kernel`, `one_pole_lp`, `sawtooth_fm`, `adsr_envelope`) |
| **`voice.c`** | How one "doll" wires those primitives into a signal chain and modulates pitch (vibrato/scoop) per sample. | `render_events()` inner loop + `_pitch_and_amp_mod()` |
| **`stream.c`** | **The main lesson: the ping-pong buffer loop** — how audio actually leaves the chip, and why you never store the song. | `mix()` / the batch render, inverted into realtime |

## The one big idea

Your Python **renders the whole song into one giant array, then plays it.**
That array would be ~20 MB; the WROOM has ~520 KB of RAM. So it can't work that way.

Instead the MCU runs **two clocks at once**:

- **The speaker's clock** — the I2S hardware pulls samples out of a small buffer
  at a fixed rate and feeds the DAC. Dedicated silicon; the CPU isn't involved
  per-sample and it never waits for you.
- **Your clock** — you keep filling *the next* small buffer just ahead of it,
  one `voice_tick()` at a time.

You hold only **2 × 256 samples (~23 ms)** in RAM ever. The song is infinite;
you just stay one buffer ahead. The run confirms this — it prints
`only 512 samples ever lived in RAM at once`.

**The single rule of embedded audio:** fill the next buffer before the hardware
drains the current one. Miss it → an audible pop (a "buffer underrun"). Our
earlier estimate put this at ~15–35% of one core, so there's comfortable slack.

## What's simplified here (so it stays legible)

- **One voice** (female "ee"), not all five. Harmony/drone/opera are the same
  `Voice` struct with different config; percussion is a sibling variant.
- **A toy note generator** (a fixed pentatonic phrase in `stream.c`) stands in
  for `generator.py` / `arrangement.py`.
- **Per-sample `sinf`/`expf`/`powf`** are left in `voice.c` for readability. On
  the real MCU you'd replace them with a recursive sine oscillator and one-state
  exp decays (≈1 multiply each) — noted in the comments where they appear.
- **`float32` throughout** — the WROOM's FPU is single-precision hardware;
  `double` would be software-emulated. (A no-FPU chip like the ESP32-C3 would
  push you to fixed-point instead.)

## From here to real firmware

1. Wire up **ESP-IDF's I2S driver** with two DMA buffers; replace the `fwrite`
   in `stream.c` with `i2s_channel_write()` (it blocks until a buffer is free —
   that block is what syncs your clock to the speaker's).
2. Put `fill_block` on **core 1** (audio only, never blocked); run the note
   generator + WiFi on **core 0**, handing note events across via a FreeRTOS
   queue.
3. Add a **MAX98357A** (I2S DAC + amp in one) — 3 wires in, speaker out.
4. Port the remaining voices + the real generator; swap per-sample transcendentals
   for table/recursive versions once you're profiling on-target.

## Wiring the MAX98357A (I2S DAC + amp in one)

This chip takes the digital I2S stream and drives a speaker directly — no
separate amp, no internal-DAC nonsense. Three signal wires match the `#define`s
in the sketches (`PIN_BCLK 26`, `PIN_LRC 25`, `PIN_DOUT 22`); any free GPIOs work
if you change the defines to match.

| MAX98357A pin | Connect to | Notes |
|---------------|-----------|-------|
| **VIN**   | 5V (or 3.3V) | 5V is louder/cleaner into 4–8 Ω. Board has its own regulator. |
| **GND**   | GND | Common ground with the ESP32. |
| **DIN**   | GPIO **22** (`PIN_DOUT`) | Serial audio data. |
| **BCLK**  | GPIO **26** (`PIN_BCLK`) | Bit clock. |
| **LRC**   | GPIO **25** (`PIN_LRC`)  | Left/right word clock. |
| **Speaker +/−** | 4–8 Ω speaker | Bridge-tied (class-D) — do **not** ground either speaker leg. |

### The two pins that trip everyone up

- **SD (shutdown / channel select)** — despite the name, this is *not* just
  on/off; the voltage on it also picks which channel plays. Leave it floating and
  the chip stays **muted**. Behavior via a resistor to GND:
  - SD held low (< 0.16 V) → **shutdown/mute**.
  - ~0.16–0.77 V (≈100 kΩ to GND, or just tie SD to VIN) → **(L+R)/2 mono** — use
    this, since we output one mono stream.
  - higher ranges → left-only / right-only.
  Simplest: **tie SD to VIN** (or through 100 kΩ) to get mono and stay un-muted.

- **GAIN** — sets amplifier gain; it reads the pin *at power-up*. Options:
  - floating → **9 dB** (a sensible default; fine to start here).
  - GAIN → GND → 12 dB · GAIN → VIN → 6 dB.
  - 100 kΩ to GND → 15 dB (loudest) · 100 kΩ to VIN → 3 dB.
  Note: many breakout boards already solder a resistor on GAIN, so check your
  board before adding your own.

### Sanity checklist if you get silence or noise
- **SD floating → muted.** Tie it to VIN.
- **Missing common ground** — the ESP32 and MAX98357A GNDs *must* be wired
  together (I2S signal levels are meaningless without a shared 0 V reference).
  If the amp is on its own supply, run a ground wire between them.
- `channel_format` in the sketch is `ONLY_LEFT` (mono) — matches SD-in-mono-mode.
- Speaker legs must both float (bridge output); grounding one can damage the chip.
