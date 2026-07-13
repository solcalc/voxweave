"""Scale + pitch lookup tables.

Everything here is a flat table or a couple lines of integer/float math, chosen
so it drops straight into C (`static const int8_t DORIAN[] = {0,2,3,5,7,9,10};`).
No music-theory library.
"""

# Semitone offsets from the root, one octave. Index by scale degree; degrees
# beyond the table length wrap up an octave (see degree_to_midi).
SCALES = {
    "major_pentatonic": (0, 2, 4, 7, 9),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
}

# Note name -> pitch class (semitones above C).
NOTE_CLASS = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11,
}


def root_midi(key: str, octave: int = 4) -> int:
    """MIDI note number for `key` in the given octave (C4 = middle C = 60)."""
    if key not in NOTE_CLASS:
        raise ValueError(f"unknown key {key!r}; choose from {sorted(NOTE_CLASS)}")
    return 12 * (octave + 1) + NOTE_CLASS[key]


def degree_to_midi(base_midi: int, scale_name: str, degree: int) -> int:
    """Map a scale degree to a MIDI note, wrapping octaves for out-of-range degrees.

    degree may be negative or larger than the scale length; it wraps and shifts
    octave accordingly. This keeps the random walk free to roam.
    """
    table = SCALES[scale_name]
    n = len(table)
    octave_shift, idx = divmod(degree, n)
    return base_midi + 12 * octave_shift + table[idx]


def snap_to_chord(degree: int, chord_tones, scale_len: int) -> int:
    """Nudge a scale-degree onto the nearest degree that is a chord tone.

    `chord_tones` are pitch-class scale degrees (0..scale_len-1). We search
    outward from `degree` (0, +1, -1, +2, -2, ...) and return the first degree
    whose pitch class (degree % scale_len) is in the chord, so a walked note lands
    on a chord tone while moving the least. Integer-only for the C port.
    """
    for step in range(scale_len + 1):
        for cand in (degree + step, degree - step):
            if cand % scale_len in chord_tones:
                return cand
    return degree  # chord_tones empty; leave the walk untouched


def midi_to_hz(m: float) -> float:
    """Equal-temperament MIDI -> frequency in Hz (A4/69 = 440 Hz)."""
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))
