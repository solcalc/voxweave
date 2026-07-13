"""Shared arrangement layer -- chord progression, phrase clock, energy contour.

Computed once per session from the SessionConstitution and read by every voice,
so the three independent lines move over a common harmonic structure instead of
just a static scale. Like generator.py this stays pure (float/int + the portable
PRNG + math), so it ports to ESP32 C. It knows nothing about audio.

The progression is *generated* (a weighted Markov walk over diatonic chord roots
that resolves to the tonic at each phrase's cadence), so structures recur but the
content never literally repeats -- endless and non-repeating by construction.
"""

import math
from dataclasses import dataclass

from scales import SCALES
from prng import voice_prng
from config import SessionConstitution


# Preference for a root move, keyed by the interval in semitones between the two
# chord roots. Perfect fifths/fourths pull hardest (functional motion), steps and
# thirds fill in, unison is discouraged so the harmony keeps moving.
_ROOT_MOVE_PREF = {
    0: 0.10, 7: 1.00, 5: 0.90, 2: 0.60, 10: 0.60,
    9: 0.50, 3: 0.45, 4: 0.45, 8: 0.40, 6: 0.30, 1: 0.25, 11: 0.25,
}


def _move_weight(table, current: int, target: int) -> float:
    """Weight of moving the chord root from degree `current` to `target`."""
    if target == current:
        return _ROOT_MOVE_PREF[0]
    semis = (table[target] - table[current]) % 12
    return _ROOT_MOVE_PREF.get(semis, 0.35)


def _pick_weighted(rng, weighted):
    """Pick an item from a list of (item, weight) using the portable PRNG."""
    total = sum(w for _, w in weighted)
    if total <= 0.0:
        return weighted[0][0]
    r = rng.rand_float() * total
    acc = 0.0
    for item, w in weighted:
        acc += w
        if r < acc:
            return item
    return weighted[-1][0]


def _next_chord(rng, table, current: int, scale_len: int) -> int:
    """Weighted Markov step to the next chord root (a scale degree)."""
    weighted = [(t, _move_weight(table, current, t)) for t in range(scale_len)]
    return _pick_weighted(rng, weighted)


def _build_chords(const: SessionConstitution, rng) -> list:
    """Per-bar chord root (a scale degree). Changes on chord-slot boundaries and
    resolves to the tonic on the last slot of each phrase (the cadence)."""
    table = SCALES[const.scale]
    scale_len = len(table)
    chord_bars = max(1, const.chord_bars)
    phrase_bars = max(1, const.phrase_bars)
    chords = []
    current = 0
    for bar in range(const.bars):
        if bar % chord_bars == 0:  # a new chord slot begins
            pos_in_phrase = bar % phrase_bars
            is_cadence_slot = pos_in_phrase + chord_bars >= phrase_bars
            if is_cadence_slot:
                current = 0  # cadence: resolve home to the tonic
            else:
                current = _next_chord(rng, table, current, scale_len)
        chords.append(current)
    return chords


def _build_energy(const: SessionConstitution, rng) -> list:
    """Per-bar intensity in [0,1]: a smooth arch over each two-phrase section plus
    a slow random drift, so the music breathes with build-ups and releases."""
    phrase_bars = max(1, const.phrase_bars)
    section_bars = phrase_bars * 2
    energy = []
    level = 0.5
    for bar in range(const.bars):
        phase = (bar % section_bars) / section_bars
        arch = 0.5 - 0.5 * math.cos(2.0 * math.pi * phase)  # 0..1, peaks mid-section
        level += (rng.rand_float() - 0.5) * 0.1             # slow long-term drift
        level = min(1.0, max(0.0, level))
        e = 0.35 + 0.5 * (0.6 * arch + 0.4 * level)
        energy.append(min(1.0, e))
    return energy


@dataclass
class Arrangement:
    """The shared structure every voice reads for the session."""

    chords: list       # per-bar chord root (scale degree)
    energy: list       # per-bar intensity 0..1
    phrase_bars: int
    chord_bars: int
    scale: str

    def _clamp(self, bar: int, seq: list):
        return seq[bar] if bar < len(seq) else seq[-1]

    def chord_root(self, bar: int) -> int:
        return self._clamp(bar, self.chords)

    def chord_tones(self, bar: int) -> tuple:
        """Chord as pitch-class scale degrees (root + stacked thirds), for snapping."""
        n = len(SCALES[self.scale])
        r = self.chord_root(bar)
        return (r % n, (r + 2) % n, (r + 4) % n)

    def energy_at(self, bar: int) -> float:
        return self._clamp(bar, self.energy)

    def phrase_index(self, bar: int) -> int:
        return bar // self.phrase_bars

    def is_cadence_bar(self, bar: int) -> bool:
        return (bar % self.phrase_bars) == self.phrase_bars - 1


def build_arrangement(const: SessionConstitution) -> Arrangement:
    """Generate the session's chord progression and energy contour."""
    rng = voice_prng(const.seed, "arrangement")
    return Arrangement(
        chords=_build_chords(const, rng),
        energy=_build_energy(const, rng),
        phrase_bars=max(1, const.phrase_bars),
        chord_bars=max(1, const.chord_bars),
        scale=const.scale,
    )


def flat_arrangement(const: SessionConstitution) -> Arrangement:
    """A neutral arrangement (static tonic, mid energy) for A/B comparison: with
    the structural voice knobs zeroed this reproduces the old free-walk sound."""
    return Arrangement(
        chords=[0] * const.bars,
        energy=[0.5] * const.bars,
        phrase_bars=max(1, const.phrase_bars),
        chord_bars=max(1, const.chord_bars),
        scale=const.scale,
    )
