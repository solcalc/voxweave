#!/usr/bin/env python3
"""CLI testbed for the singing-doll generative music algorithm.

Examples:
    python testbed.py --voice melody --key C --scale dorian --seed 42
    python testbed.py --voice all --seed 42
    python testbed.py --voice all --scale lydian --key G --tempo 50 --play

Renders are written to output/ with descriptive filenames including seed, key,
scale, and voice, so past runs can be A/B compared without regenerating.
"""

import argparse
import os

import generator
import synth
import wav_io
from arrangement import build_arrangement, flat_arrangement
from config import SessionConstitution, default_voices
from scales import SCALES, NOTE_CLASS


def _filename(const: SessionConstitution, voice: str) -> str:
    return f"seed{const.seed}_{const.key}-{const.scale}_{voice}.wav"


def _render_voice(const, cfg, arr):
    events = generator.generate_voice(const, cfg, arr)
    return synth.render_events(events, const, cfg)


def _flatten_voice(cfg):
    """Zero the structural knobs so a voice reverts to the old free random walk
    (for A/B against the arrangement). Pair with a flat_arrangement()."""
    cfg.chord_lock = 0.0
    cfg.subdivide = 0.0
    cfg.energy_response = 0.0
    cfg.motif_len = 0
    cfg.beat_accent = (1.0,)
    cfg.answer_parity = -1


def _print_progression(const, arr):
    print(f"progression ({const.key} {const.scale}, "
          f"phrase={const.phrase_bars} chord={const.chord_bars} bars):")
    for bar in range(const.bars):
        mark = " <cadence" if arr.is_cadence_bar(bar) else ""
        print(f"  bar {bar:3d}  chord deg {arr.chord_root(bar)}  "
              f"tones {arr.chord_tones(bar)}  energy {arr.energy_at(bar):.2f}{mark}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--voice", choices=["melody", "harmony", "drone", "beat", "all"],
                   default="all", help="which voice to render (default: all)")
    p.add_argument("--key", default="C", choices=sorted(NOTE_CLASS),
                   help="root key (default: C)")
    p.add_argument("--scale", default="dorian", choices=sorted(SCALES),
                   help="scale (default: dorian)")
    p.add_argument("--tempo", type=int, default=60, help="BPM (default: 60)")
    p.add_argument("--seed", type=int, default=42, help="PRNG seed (default: 42)")
    p.add_argument("--bars", type=int, default=16, help="bar count (default: 16)")
    p.add_argument("--sample-rate", type=int, default=44100, help="Hz (default: 44100)")
    p.add_argument("--density-melody", type=float, default=None)
    p.add_argument("--density-harmony", type=float, default=None)
    p.add_argument("--density-drone", type=float, default=None)
    p.add_argument("--vowel-melody", default=None)
    p.add_argument("--vowel-harmony", default=None)
    p.add_argument("--vowel-drone", default=None)
    p.add_argument("--phrase-bars", type=int, default=4, help="bars per phrase (default: 4)")
    p.add_argument("--chord-bars", type=int, default=2, help="bars per chord (default: 2)")
    p.add_argument("--flat", action="store_true",
                   help="bypass the arrangement (old free-walk sound) for A/B")
    p.add_argument("--print-progression", action="store_true",
                   help="print the generated chord progression + energy and exit")
    p.add_argument("--out-dir", default="output", help="output folder (default: output/)")
    p.add_argument("--play", action="store_true", help="also play the render aloud")
    args = p.parse_args()

    const = SessionConstitution(
        key=args.key, scale=args.scale, tempo=args.tempo,
        seed=args.seed, bars=args.bars, sample_rate=args.sample_rate,
        phrase_bars=args.phrase_bars, chord_bars=args.chord_bars,
    )

    arr = flat_arrangement(const) if args.flat else build_arrangement(const)
    if args.print_progression:
        _print_progression(const, arr)
        return

    # Build voices from defaults, applying per-voice overrides.
    voices = default_voices()
    overrides = {
        "melody": (args.density_melody, args.vowel_melody),
        "harmony": (args.density_harmony, args.vowel_harmony),
        "drone": (args.density_drone, args.vowel_drone),
    }
    for name, (dens, vow) in overrides.items():
        if dens is not None:
            voices[name].density = dens
        if vow is not None:
            voices[name].vowel = vow

    if args.flat:
        for cfg in voices.values():
            _flatten_voice(cfg)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.voice == "all":
        buffers = []
        for name in ("melody", "harmony", "drone", "beat"):
            buf = _render_voice(const, voices[name], arr)
            buffers.append(buf)
            path = os.path.join(args.out_dir, _filename(const, name))
            wav_io.write_wav(path, buf, const.sample_rate)
            print(f"wrote {path}  ({len(buf) / const.sample_rate:.1f}s)")
        mixed = synth.mix(buffers)
        mix_path = os.path.join(args.out_dir, _filename(const, "all"))
        wav_io.write_wav(mix_path, mixed, const.sample_rate)
        print(f"wrote {mix_path}  ({len(mixed) / const.sample_rate:.1f}s)")
        final = mixed
    else:
        cfg = voices[args.voice]
        final = _render_voice(const, cfg, arr)
        path = os.path.join(args.out_dir, _filename(const, args.voice))
        wav_io.write_wav(path, final, const.sample_rate)
        print(f"wrote {path}  ({len(final) / const.sample_rate:.1f}s)")

    if args.play:
        wav_io.play(final, const.sample_rate)


if __name__ == "__main__":
    main()
