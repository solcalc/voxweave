"""WAV output via the stdlib `wave` module + optional direct playback.

Kept deliberately dependency-light: 16-bit PCM through the standard library, so
nothing here needs libsndfile or scipy. Playback is best-effort via sounddevice.
"""

import wave

import numpy as np


def _to_int16(samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2")


def write_wav(path: str, samples: np.ndarray, sr: int) -> None:
    """Write a mono float buffer (~[-1, 1]) to a 16-bit PCM WAV file."""
    pcm = _to_int16(samples)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)          # 16-bit
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def play(samples: np.ndarray, sr: int) -> None:
    """Play a mono float buffer directly. No-op with a note if sounddevice is missing."""
    try:
        import sounddevice as sd
    except Exception as exc:  # ImportError, or PortAudio not installed
        print(f"[play] sounddevice unavailable ({exc}); skipping direct playback.")
        return
    sd.play(np.clip(samples, -1.0, 1.0).astype(np.float32), sr)
    sd.wait()
