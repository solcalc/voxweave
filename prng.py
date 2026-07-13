"""Portable deterministic PRNG.

Deliberately tiny and integer-only so it ports 1:1 to C on the ESP32. The dolls
will receive a shared 32-bit seed over ESP-NOW and must reproduce the exact same
sequence -- that only works if every doll (and this testbed) runs the *same*
generator. So we do NOT use numpy's global random state or Python's `random`;
we use a plain xorshift32.

Reference C port:

    uint32_t s;                       // state, must be non-zero
    uint32_t next_u32(void) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return s;
    }
"""

_UINT32_MASK = 0xFFFFFFFF


class PRNG:
    """xorshift32 deterministic pseudo-random generator."""

    def __init__(self, seed: int):
        # State must be non-zero; fold seed and force a safe non-zero value.
        s = seed & _UINT32_MASK
        if s == 0:
            s = 0x1234_5678
        self.state = s

    def next_u32(self) -> int:
        """Advance state and return a 32-bit unsigned int."""
        s = self.state
        s ^= (s << 13) & _UINT32_MASK
        s ^= s >> 17
        s ^= (s << 5) & _UINT32_MASK
        s &= _UINT32_MASK
        self.state = s
        return s

    def rand_float(self) -> float:
        """Uniform float in [0, 1)."""
        # 24 bits of mantissa is plenty and matches a float32 C port.
        return (self.next_u32() >> 8) / float(1 << 24)

    def randint(self, lo: int, hi: int) -> int:
        """Uniform integer in [lo, hi] inclusive."""
        span = hi - lo + 1
        return lo + (self.next_u32() % span)

    def choice(self, seq):
        """Pick one element of a non-empty sequence."""
        return seq[self.next_u32() % len(seq)]

    def chance(self, p: float) -> bool:
        """Return True with probability p (0..1)."""
        return self.rand_float() < p


# Fixed per-voice salts XORed into the base seed so each voice has an
# independent stream, yet a voice rendered alone is identical to that same
# voice rendered inside the full mix. Keep these stable to keep renders stable.
VOICE_SALT = {
    "melody": 0x00000000,
    "harmony": 0x9E3779B9,  # golden-ratio constant, good bit spread
    "drone": 0x85EBCA6B,
    "beat": 0x27D4EB2F,     # beatbox voice gets its own stream (not melody's salt 0)
    # The arrangement (chords/phrase/energy) is a session-wide stream, independent
    # of any voice yet reproducible from the same seed. It gets its own salt so the
    # shared harmonic structure is stable no matter which voices are rendered.
    "arrangement": 0xC2B2AE35,
}


def voice_prng(base_seed: int, voice: str) -> PRNG:
    """Derive a per-voice PRNG from the shared session seed."""
    salt = VOICE_SALT.get(voice, 0)
    return PRNG((base_seed ^ salt) & _UINT32_MASK)
