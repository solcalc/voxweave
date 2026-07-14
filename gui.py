#!/usr/bin/env python3
"""Desktop GUI for the singing-doll generative music testbed (Dear PyGui).

A thin shell over the existing pipeline -- it edits every session/voice parameter,
toggles voices on/off, renders via generate_voice -> render_events -> mix, and
draws each voice as a read-only piano roll. Nothing in the generator/synth core
is modified; this module only reads from it.

    python gui.py

Live mode: tick the "Live" checkbox, then any parameter change triggers a background
re-render. When the new buffer is ready it swaps in at the same proportional position
in the song so playback continues uninterrupted. Loop mode restarts from bar 0 when
the piece ends.
"""

import ast
import dataclasses
import json
import math
import os
import random
import threading
import time

import numpy as np
import dearpygui.dearpygui as dpg

import generator
import synth
import wav_io
from arrangement import build_arrangement, flat_arrangement
from config import SessionConstitution, VoiceConfig, VOWEL_FORMANTS, default_voices
from scales import NOTE_CLASS, SCALES

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except Exception:
    _SD_AVAILABLE = False

VOICE_NAMES = ("melody", "harmony", "drone", "beat")
VOICE_COLORS = {
    "melody": (90, 185, 255),
    "harmony": (255, 185, 90),
    "drone": (150, 235, 150),
    "beat": (235, 120, 200),
}

CANVAS_W = 1400
CANVAS_H = 560
MARGIN = 40

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui_settings.json")

# How long after the last param tweak before a live re-render fires (seconds).
LIVE_DEBOUNCE_SEC = 0.6

SESSION_INT_FIELDS = (
    "tempo", "seed", "bars", "beats_per_bar",
    "sample_rate", "phrase_bars", "chord_bars",
)

SESSION_HELP = {
    "key": "The 'home' note the whole piece is built around, like C or G. Higher "
           "letters sound higher. Pick one from the list.",
    "scale": "The family of notes the music draws from -- each choice has its own "
             "mood. Pick one from the list.",
    "tempo": "Speed of the music: higher = faster. A whole number of beats per "
             "minute; this gentle style sounds best slow (about 40-90).",
    "seed": "A number that locks in one exact random result -- the same number "
            "always makes the identical piece. Any whole number.",
    "seed_random": "When on, a brand-new random number is used every time you press "
                   "Generate, so you get different music each time. The box updates "
                   "to show the number it used. On/off.",
    "bars": "How long the piece is, counted in measures ('bars'). More bars = longer "
            "music. A whole number.",
    "beats_per_bar": "How many beats are in each measure -- 4 is the usual count you'd "
                     "tap your foot to. A whole number.",
    "sample_rate": "Audio quality (detail per second). 44100 is normal CD quality and "
                   "you rarely need to change it. A whole number.",
    "phrase_bars": "How many bars make up one musical 'sentence' before it reaches a "
                   "natural resting point. A whole number.",
    "chord_bars": "How many bars each background chord lasts before it changes. "
                  "Smaller = the harmony shifts more often. A whole number.",
    "flat": "Turns off the automatic song structure (chords, build-ups, back-and-forth "
            "between voices) so the voices just wander freely. Good for comparison. "
            "On/off.",
}

VOICE_ENABLED_HELP = "Turn this voice on or off. When off, you won't hear it at all. On/off."

VOICE_HELP = {
    "vowel": "Which vowel this voice 'sings' (like 'eee', 'oh', or 'oo'), which changes "
             "its character. Pick one from the list.",
    "octave": "How high or low this voice sits. 0 is its normal spot; +1 jumps it up "
              "higher, -1 drops it lower. A whole number.",
    "density": "How busy this voice is -- how often it sings instead of resting. "
               "0 = silent, 1 = plays almost constantly. A number from 0 to 1.",
    "walk_steps": "How far the tune can step between notes on each move -- small numbers "
                  "sound smooth, bigger ones sound jumpier. A list of numbers in "
                  "parentheses, e.g. (-2, -1, 1, 2).",
    "start_degree": "Which note of the scale the voice starts on (0 = the home note). "
                    "Just nudges its starting pitch up or down. A whole number.",
    "degree_pool": "(Low 'drone' voice only) The set of low notes it's allowed to hold. "
                   "A list of numbers in parentheses.",
    "change_bars_min": "(Drone only) The fewest bars it holds a note before it's allowed "
                       "to switch. A whole number.",
    "change_bars_max": "(Drone only) The most bars it will hold a note before switching. "
                       "A whole number.",
    "attack": "How quickly each note fades in at its start. Small = it appears instantly; "
              "larger = it swells in gently. In seconds.",
    "decay": "How long each note takes to fade away after it's sung. Bigger = longer, "
             "more lingering tails. In seconds.",
    "note_frac": "How much of its time each note holds before it starts fading out. "
                 "Higher sounds more smooth and connected. A number from 0 to 1.",
    "amp": "How loud this voice is in the mix. 0 = silent, 1 = full volume. A number "
           "from 0 to 1.",
    "chord_lock": "How strongly this voice sticks to notes that fit the current "
                  "background harmony (sounding 'in tune together') versus wandering on "
                  "its own. 0 = free, 1 = always fits. A number from 0 to 1.",
    "beat_accent": "How much each beat of the bar is emphasized -- higher numbers make "
                   "that beat more likely to play, creating a pulse. A list of numbers, "
                   "one per beat, e.g. (1.3, 0.7, 1.1, 0.6).",
    "subdivide": "How often a beat splits into two quicker notes, adding little "
                 "flourishes. 0 = never, 1 = often. A number from 0 to 1.",
    "motif_len": "The length of a short musical idea this voice keeps returning to. "
                 "0 turns off repeating ideas. A whole number.",
    "motif_bars": "How many bars before that repeating idea gets a fresh variation. "
                  "A whole number.",
    "energy_response": "How much this voice reacts to the music's build-ups and calmer "
                       "moments (getting louder/busier in the intense parts). 0 = ignores "
                       "them, 1 = follows closely. A number from 0 to 1.",
    "answer_parity": "Sets up call-and-response between voices -- whether this one sings "
                     "during the 'question' or the 'answer'. Use 0 or 1 to trade off with "
                     "another voice, or -1 to always sing. A whole number (-1, 0, or 1).",
    "vibrato_hz": "Vibrato is the gentle wobble a singer adds to a held note. This sets "
                  "how fast that wobble is. Higher = faster wobble.",
    "vibrato_semitones": "How wide the vibrato wobble is -- how far the pitch drifts up "
                         "and down. Small numbers are subtle.",
    "vibrato_delay": "How long a note waits before the wobble kicks in (real singers hold "
                     "steady first, then add it). In seconds.",
    "vibrato_ramp": "Once the wobble starts, how long it takes to grow from none to full. "
                    "In seconds.",
    "tremolo": "A gentle rise-and-fall in loudness that rides along with the vibrato "
               "wobble. 0 = none, higher = more noticeable. A number from 0 to 1.",
    "scoop_semitones": "A little slide up into each note from just below, like a singer "
                       "'scooping' up to the pitch. How far it slides from; 0 = no slide.",
    "scoop_time": "How quick that scoop-up slide is. Smaller = snappier. In seconds.",
    "jitter_cents": "Tiny random pitch imperfections between notes so it sounds human "
                    "instead of robotic. Bigger = more wobble. (Measured in 'cents', a "
                    "very small pitch amount.)",
    "shimmer": "Tiny random changes in loudness from note to note, for a more natural "
               "feel. Keep it small. A small number (around 0 to 0.2).",
    "breath": "How much breathy, airy noise is mixed in, like hearing a singer's breath. "
              "0 = clean, higher = breathier. A number from 0 to 1.",
    "tilt_hz": "Softens the harsh, buzzy edge of the sound to make it smoother and "
               "warmer. Lower numbers = mellower; 0 turns it off.",
}

# --------------------------------------------------------------------------- #
# Live state.
# --------------------------------------------------------------------------- #

# Guards state["mixed"] and state["play_sample"] during buffer swaps.
# The audio callback runs on a PortAudio thread, so we need this to prevent
# a torn read while we swap in a freshly rendered buffer.
_buf_lock = threading.Lock()

state = {
    "voices": default_voices(),
    "const": None,
    "events": {},
    "mixed": None,
    "geom": None,
    # --- background generation ---
    "gen_running": False,
    "gen_cancel": False,
    "gen_progress": 0.0,
    "gen_status": "",
    "gen_result": None,        # (const, events, voice_buffers, mixed)
    "arr": None,               # last Arrangement, reused by per-voice live regen
    "voice_buffers": {},       # {name: np.ndarray}, per-voice rendered audio
    # --- streaming playback ---
    "playing": False,
    "play_sample": 0,          # current read position (samples) into state["mixed"]
    "sd_stream": None,         # open sounddevice.OutputStream, or None
    "mixed_seg_len": 0,        # musical length (samples) of state["mixed"]; buffer runs
                               # longer for decay tails, but a segment ENDS here
    "next_seg_len": 0,         # musical length of the queued next_buf
    "next_tail_merged": False, # has the current decay tail been overlap-added into next?
    # --- volume ---
    "master_vol": 1.0,         # applied as a real-time gain in the audio callback
    "voice_vol": {n: 1.0 for n in VOICE_NAMES},  # per-voice gains applied at mix time
    # --- live mode ---
    "live_mode": False,
    "param_snapshot": None,    # widget-value dict at last regen (change baseline)
    "last_change_t": 0.0,      # perf_counter when last param change was detected
    # --- endless mode ---
    "endless": False,
    "endless_gen_running": False,
    "endless_swapped": False,  # set by audio thread when it consumes next_buf
    "next_buf": None,          # pre-rendered next segment (audio)
    "next_events": {},
    "next_const": None,
    "next_voice_buffers": {},
    "_roll_frame": 0,          # throttle piano-roll redraws during playback
}


# --------------------------------------------------------------------------- #
# Flatten: zero the structural knobs (parity with testbed.py --flat).
# --------------------------------------------------------------------------- #
def _flatten_voice(cfg: VoiceConfig) -> None:
    cfg.chord_lock = 0.0
    cfg.subdivide = 0.0
    cfg.energy_response = 0.0
    cfg.motif_len = 0
    cfg.beat_accent = (1.0,)
    cfg.answer_parity = -1


# --------------------------------------------------------------------------- #
# Widget tag helpers.
# --------------------------------------------------------------------------- #
def _voice_tag(name: str, field: str) -> str:
    return f"v_{name}_{field}"


def _enabled_tag(name: str) -> str:
    return f"v_{name}_enabled"


def _status(msg: str) -> None:
    dpg.set_value("status_text", msg)


# --------------------------------------------------------------------------- #
# Read widgets -> config objects.
# --------------------------------------------------------------------------- #
def _read_const() -> SessionConstitution:
    kwargs = {name: int(dpg.get_value(f"session_{name}")) for name in SESSION_INT_FIELDS}
    return SessionConstitution(
        key=dpg.get_value("session_key"),
        scale=dpg.get_value("session_scale"),
        **kwargs,
    )


def _apply_voice_widgets() -> None:
    """Push every voice widget value back into its VoiceConfig object."""
    for name in VOICE_NAMES:
        cfg = state["voices"][name]
        for f in dataclasses.fields(VoiceConfig):
            if f.name == "name":
                continue
            raw = dpg.get_value(_voice_tag(name, f.name))
            cur = getattr(cfg, f.name)
            if isinstance(cur, tuple):
                parsed = ast.literal_eval(raw)
                setattr(cfg, f.name, tuple(parsed) if isinstance(parsed, (list, tuple)) else (parsed,))
            elif isinstance(cur, bool):
                setattr(cfg, f.name, bool(raw))
            elif isinstance(cur, int):
                setattr(cfg, f.name, int(raw))
            elif isinstance(cur, float):
                setattr(cfg, f.name, float(raw))
            else:
                setattr(cfg, f.name, str(raw))


# --------------------------------------------------------------------------- #
# Core actions.
# --------------------------------------------------------------------------- #
def _generate_worker(const, flat, enabled) -> None:
    """Runs off the UI thread: builds the arrangement, renders each enabled voice,
    reporting progress and honouring a cancel request between voices."""
    try:
        arr = flat_arrangement(const) if flat else build_arrangement(const)
        state["arr"] = arr
        steps = len(enabled) + 1
        state["gen_progress"] = 1.0 / steps
        events, voice_buffers, buffers = {}, {}, []
        for i, (name, cfg) in enumerate(enabled):
            if state["gen_cancel"]:
                state["gen_status"] = "cancelled"
                state["gen_running"] = False
                return
            state["gen_status"] = f"rendering {name}..."
            evs = generator.generate_voice(const, cfg, arr)
            events[name] = evs
            buf = synth.render_events(evs, const, cfg)
            voice_buffers[name] = buf   # store unscaled; per-voice gain applied at mix
            buffers.append(buf * state["voice_vol"].get(name, 1.0))
            state["gen_progress"] = (i + 2) / steps
        mixed = synth.mix(buffers) if buffers else None
        state["gen_result"] = (const, events, voice_buffers, mixed)
    except Exception as exc:
        state["gen_status"] = f"error: {exc}"
    finally:
        state["gen_running"] = False


def on_generate() -> None:
    if state["gen_running"]:
        return
    if dpg.get_value("session_seed_random"):
        dpg.set_value("session_seed", random.randint(0, 2**31 - 1))
    try:
        const = _read_const()
        _apply_voice_widgets()
    except (ValueError, SyntaxError) as exc:
        _status(f"invalid parameter: {exc}")
        return

    flat = dpg.get_value("session_flat")
    enabled = []
    for name in VOICE_NAMES:
        if not dpg.get_value(_enabled_tag(name)):
            continue
        cfg = state["voices"][name]
        if flat:
            _flatten_voice(cfg)
        enabled.append((name, cfg))

    if not enabled:
        state["const"] = const
        state["events"] = {}
        with _buf_lock:
            state["mixed"] = None
            state["play_sample"] = 0
        draw_piano_roll()
        _status("no voices enabled")
        return

    state.update(gen_running=True, gen_cancel=False, gen_progress=0.0,
                 gen_status="generating...", gen_result=None)
    threading.Thread(target=_generate_worker, args=(const, flat, enabled),
                     daemon=True).start()


def on_cancel_gen() -> None:
    if state["gen_running"]:
        state["gen_cancel"] = True
        _status("cancelling...")


# --------------------------------------------------------------------------- #
# Streaming playback.
# --------------------------------------------------------------------------- #
def _audio_cb(outdata, frames, time_info, status):
    """PortAudio callback: feeds chunks from state["mixed"] into the audio driver.

    In endless mode a segment ENDS at its musical boundary (mixed_seg_len), not at
    the physical end of the buffer -- the extra samples past that point are the decay
    tail, which has already been overlap-added into the head of the next segment
    (see frame_update). So we swap at the boundary and the ring-out carries over
    seamlessly, keeping audio in lockstep with the piano roll."""
    with _buf_lock:
        buf = state["mixed"]
        pos = state["play_sample"]
        playing = state["playing"]
        seg_len = state["mixed_seg_len"]

    if buf is None or not playing:
        outdata[:] = 0
        return

    buf_len = len(buf)
    if seg_len <= 0 or seg_len > buf_len:
        seg_len = buf_len
    out = np.zeros(frames, dtype=np.float32)
    filled = 0
    swapped = False

    while filled < frames:
        # Swap at the musical boundary once the next segment is ready.
        if (state["endless"] and not swapped and pos >= seg_len
                and state["next_buf"] is not None):
            nb = state["next_buf"]
            new_seg = state["next_seg_len"]
            with _buf_lock:
                state["mixed"] = nb
                state["mixed_seg_len"] = new_seg
                state["next_buf"] = None
                state["endless_swapped"] = True
            buf = nb
            buf_len = len(buf)
            seg_len = new_seg if (0 < new_seg <= buf_len) else buf_len
            pos = 0
            swapped = True
            continue

        avail = buf_len - pos
        if avail <= 0:
            if state["endless"]:
                pos = 0        # physical end w/o a ready next: loop to avoid silence
                continue
            else:
                state["playing"] = False
                break

        # In endless mode, don't read past the musical boundary if we can swap there.
        limit = avail
        if state["endless"] and pos < seg_len and state["next_buf"] is not None:
            limit = min(limit, seg_len - pos)
        take = min(limit, frames - filled)
        out[filled : filled + take] = buf[pos : pos + take].astype(np.float32)
        filled += take
        pos += take

    out *= state["master_vol"]        # real-time master gain
    outdata[:, 0] = out
    with _buf_lock:
        state["play_sample"] = pos if state["playing"] else 0


def _ensure_stream(sr: int) -> bool:
    """Open a persistent OutputStream at the given sample rate.
    Returns True on success, False if sounddevice is unavailable."""
    if not _SD_AVAILABLE:
        return False
    existing = state["sd_stream"]
    if existing is not None:
        if existing.active and int(existing.samplerate) == sr:
            return True
        existing.close()
        state["sd_stream"] = None
    try:
        stream = sd.OutputStream(
            samplerate=sr,
            channels=1,
            dtype="float32",
            callback=_audio_cb,
            blocksize=1024,
        )
        stream.start()
        state["sd_stream"] = stream
        return True
    except Exception as exc:
        _status(f"audio stream error: {exc}")
        return False


def _close_stream() -> None:
    stream = state["sd_stream"]
    if stream is not None:
        try:
            stream.close()
        except Exception:
            pass
        state["sd_stream"] = None


# --------------------------------------------------------------------------- #
# Volume.
# --------------------------------------------------------------------------- #
def _remix_current() -> None:
    """Rebuild state["mixed"] from the stored (unscaled) per-voice buffers using the
    current per-voice gains, preserving playback position. Cheap; no re-render."""
    vb = state["voice_buffers"]
    if not vb:
        return
    buffers = [vb[n] * state["voice_vol"].get(n, 1.0)
               for n in VOICE_NAMES if n in vb]
    mixed = synth.mix(buffers) if buffers else None
    with _buf_lock:
        old = state["mixed"]
        if old is not None and mixed is not None and len(old) > 0:
            frac = state["play_sample"] / len(old)
            state["mixed"] = mixed
            state["play_sample"] = min(int(frac * len(mixed)), len(mixed) - 1)
        else:
            state["mixed"] = mixed


def _on_master_vol(sender, app_data) -> None:
    state["master_vol"] = float(app_data)


def _on_voice_vol(sender, app_data, user_data) -> None:
    state["voice_vol"][user_data] = float(app_data)
    _remix_current()
    # Keep any queued endless segment in sync too, so it swaps in at the new levels.
    # Re-mixing drops the previously merged decay tail; let it re-merge next frame.
    nvb = state["next_voice_buffers"]
    if nvb and state["next_buf"] is not None:
        buffers = [nvb[n] * state["voice_vol"].get(n, 1.0)
                   for n in VOICE_NAMES if n in nvb]
        state["next_buf"] = synth.mix(buffers) if buffers else None
        state["next_tail_merged"] = False


def _sync_volumes_from_widgets() -> None:
    """Pull slider values into state (set_value does not fire callbacks)."""
    if dpg.does_item_exist("master_vol"):
        state["master_vol"] = float(dpg.get_value("master_vol"))
    for n in VOICE_NAMES:
        tag = f"vol_{n}"
        if dpg.does_item_exist(tag):
            state["voice_vol"][n] = float(dpg.get_value(tag))


def on_play() -> None:
    if state["mixed"] is None:
        _status("generate first")
        return
    if not _ensure_stream(state["const"].sample_rate):
        _status("playback unavailable (sounddevice not installed)")
        return
    with _buf_lock:
        state["play_sample"] = 0
    state["playing"] = True
    _status("playing...")


def on_stop() -> None:
    state["playing"] = False
    with _buf_lock:
        state["play_sample"] = 0
    _status("stopped")


def on_export() -> None:
    if state["mixed"] is None:
        _status("generate first")
        return
    const = state["const"]
    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", f"seed{const.seed}_{const.key}-{const.scale}_gui.wav")
    wav_io.write_wav(path, state["mixed"], const.sample_rate)
    _status(f"wrote {path}")


# --------------------------------------------------------------------------- #
# Persistence: remember every widget value between sessions.
# --------------------------------------------------------------------------- #
def _setting_tags():
    """Every widget tag whose value is saved/restored/reset."""
    tags = ["session_key", "session_scale", "session_seed_random", "session_flat",
            "session_live", "session_endless"]
    tags += [f"session_{n}" for n in SESSION_INT_FIELDS]
    for name in VOICE_NAMES:
        tags.append(_enabled_tag(name))
        for f in dataclasses.fields(VoiceConfig):
            if f.name != "name":
                tags.append(_voice_tag(name, f.name))
    return tags


def _volume_tags():
    """Volume slider tags: saved/restored, but kept out of the live-mode change
    snapshot so moving a fader re-mixes instantly without a re-render."""
    return ["master_vol"] + [f"vol_{n}" for n in VOICE_NAMES]


def save_settings() -> None:
    tags = _setting_tags() + _volume_tags()
    data = {tag: dpg.get_value(tag) for tag in tags if dpg.does_item_exist(tag)}
    try:
        with open(SETTINGS_PATH, "w") as fh:
            json.dump(data, fh, indent=2)
    except OSError as exc:
        print(f"[gui] could not save settings: {exc}")


def load_settings() -> None:
    if not os.path.exists(SETTINGS_PATH):
        return
    try:
        with open(SETTINGS_PATH) as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"[gui] could not load settings: {exc}")
        return
    for tag, val in data.items():
        if dpg.does_item_exist(tag):
            try:
                dpg.set_value(tag, val)
            except Exception:
                pass


def _default_settings() -> dict:
    """The value every widget should hold in its factory-default state."""
    const = SessionConstitution()
    voices = default_voices()
    out = {
        "session_key": const.key,
        "session_scale": const.scale,
        "session_seed_random": True,
        "session_flat": False,
        "session_live": True,
        "session_endless": True,
    }
    for tag in _volume_tags():
        out[tag] = 1.0
    for n in SESSION_INT_FIELDS:
        out[f"session_{n}"] = getattr(const, n)
    for name in VOICE_NAMES:
        out[_enabled_tag(name)] = True
        cfg = voices[name]
        for f in dataclasses.fields(VoiceConfig):
            if f.name == "name":
                continue
            v = getattr(cfg, f.name)
            out[_voice_tag(name, f.name)] = repr(v) if isinstance(v, tuple) else v
    return out


def on_reset_defaults() -> None:
    state["voices"] = default_voices()
    for tag, val in _default_settings().items():
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, val)
    _sync_volumes_from_widgets()
    _remix_current()
    _status("reset to defaults")


# --------------------------------------------------------------------------- #
# Live mode helpers.
# --------------------------------------------------------------------------- #
def _param_snapshot() -> dict:
    """Snapshot all controllable widget values for change detection."""
    return {tag: dpg.get_value(tag)
            for tag in _setting_tags()
            if dpg.does_item_exist(tag)}


def _changed_voices(old_snap: dict, new_snap: dict):
    """
    Compare two snapshots. Returns the set of voice names whose params changed,
    or None if any session-level param changed (which requires a full regen).
    """
    changed = set()
    for tag, old_val in old_snap.items():
        if new_snap.get(tag) == old_val:
            continue
        matched = None
        for vname in VOICE_NAMES:
            if tag.startswith(f"v_{vname}_"):
                matched = vname
                break
        if matched is not None:
            changed.add(matched)
        else:
            return None  # session param changed
    return changed


def _revoice_worker(names: set) -> None:
    """Re-render only the named voices from their existing note events, then re-mix.
    Session params (key, scale, tempo …) and note sequences are unchanged; only
    the synth parameters (amp, vowel, vibrato, etc.) are re-applied."""
    try:
        const = state["const"]
        voice_buffers = dict(state["voice_buffers"])  # shallow copy
        events = state["events"]

        n_voices = len(names)
        for idx, name in enumerate(names):
            state["gen_progress"] = (idx + 1) / (n_voices + 1)
            state["gen_status"] = f"updating {name}..."
            if not dpg.get_value(_enabled_tag(name)):
                voice_buffers.pop(name, None)
                continue
            cfg = state["voices"][name]
            evs = events.get(name, [])
            if evs:
                voice_buffers[name] = synth.render_events(evs, const, cfg)

        buffers = [voice_buffers[n] * state["voice_vol"].get(n, 1.0)
                   for n in VOICE_NAMES if n in voice_buffers]
        mixed = synth.mix(buffers) if buffers else None
        state["gen_result"] = (const, events, voice_buffers, mixed)
    except Exception as exc:
        state["gen_status"] = f"error: {exc}"
    finally:
        state["gen_running"] = False


def _endless_gen_worker(const, flat, enabled) -> None:
    """Generate the next endless segment into state["next_buf"] without interrupting playback."""
    try:
        arr = flat_arrangement(const) if flat else build_arrangement(const)
        events, voice_buffers, buffers = {}, {}, []
        for name, cfg in enabled:
            evs = generator.generate_voice(const, cfg, arr)
            events[name] = evs
            buf = synth.render_events(evs, const, cfg)
            voice_buffers[name] = buf   # store unscaled; per-voice gain applied at mix
            buffers.append(buf * state["voice_vol"].get(name, 1.0))
        mixed = synth.mix(buffers) if buffers else None
        # Publish metadata first, then next_buf last: the audio thread swaps on
        # next_buf, so everything it needs must already be in place when it sees it.
        state["next_events"] = events
        state["next_const"] = const
        state["next_voice_buffers"] = voice_buffers
        state["next_seg_len"] = int(const.total_seconds * const.sample_rate)
        state["next_tail_merged"] = False
        state["next_buf"] = mixed
    except Exception as exc:
        print(f"[endless] generation error: {exc}")
    finally:
        state["endless_gen_running"] = False


def _invalidate_next() -> None:
    """Drop any queued/pre-rendered next endless segment so preload rebuilds it."""
    state["next_buf"] = None
    state["next_events"] = {}
    state["next_const"] = None
    state["next_voice_buffers"] = {}
    state["next_seg_len"] = 0
    state["next_tail_merged"] = False


def _trigger_endless_preload() -> None:
    """Kick off background generation of the next segment with a fresh random seed."""
    const = state["const"]
    if const is None or state["endless_gen_running"]:
        return
    next_const = dataclasses.replace(const, seed=random.randint(0, 2**31 - 1))
    flat = dpg.get_value("session_flat")
    enabled = [(name, state["voices"][name]) for name in VOICE_NAMES
               if dpg.get_value(_enabled_tag(name))]
    if not enabled:
        return
    state["endless_gen_running"] = True
    threading.Thread(target=_endless_gen_worker, args=(next_const, flat, enabled),
                     daemon=True).start()


def _trigger_live_regen(old_snap: dict, new_snap: dict) -> None:
    """Debounce fired: re-render only what changed, or do a full regen for session edits."""
    state["last_change_t"] = 0.0
    state["param_snapshot"] = new_snap  # new baseline; suppress re-trigger

    _apply_voice_widgets()
    changed = _changed_voices(old_snap, new_snap)

    if changed is None:
        # Session params changed (key, tempo, bars …): full regen with new seed if needed.
        on_generate()
        state["param_snapshot"] = _param_snapshot()  # absorb seed widget update
    elif changed:
        # Only specific voice params changed: re-render just those voices.
        state.update(gen_running=True, gen_cancel=False, gen_progress=0.0,
                     gen_status="updating...", gen_result=None)
        threading.Thread(target=_revoice_worker, args=(changed,), daemon=True).start()


# --------------------------------------------------------------------------- #
# Piano roll.
# --------------------------------------------------------------------------- #
def draw_piano_roll(current_time: float | None = None) -> None:
    """Draw the piano roll.

    current_time=None  → static view of the whole piece (called on generate / stop).
    current_time=float → scrolling view; notes move left past a fixed playhead.
    """
    dpg.delete_item("piano_roll", children_only=True)
    const = state["const"]
    events = state["events"]

    plot_x0, plot_y0 = MARGIN, MARGIN
    plot_w = CANVAS_W - 2 * MARGIN
    plot_h = CANVAS_H - 2 * MARGIN

    dpg.draw_rectangle((plot_x0, plot_y0), (plot_x0 + plot_w, plot_y0 + plot_h),
                       fill=(28, 28, 32), color=(70, 70, 80), parent="piano_roll")

    all_events = [ev for evs in events.values() for ev in evs]
    if not all_events or const is None:
        state["geom"] = None
        dpg.draw_text((plot_x0 + 12, plot_y0 + 12), "no notes -- press Generate",
                      size=18, color=(160, 160, 160), parent="piano_roll")
        return

    total_secs = max(const.total_seconds, 1e-6)
    sec_per_bar = const.sec_per_bar

    # In scrolling mode, chain the already-rendered next segment right after the
    # current one so notes flow past the playhead with no gap at the boundary.
    # Each entry: (events_dict, time_offset).
    segments = [(events, 0.0)]
    next_events = state["next_events"]
    if current_time is not None and next_events:
        segments.append((next_events, total_secs))

    pitches = [ev.midi for evs, _ in segments for lst in evs.values() for ev in lst]
    minp, maxp = min(pitches) - 2, max(pitches) + 2
    span = max(maxp - minp, 1)
    row_h = plot_h / (span + 1)

    if current_time is None:
        # Static: entire piece fits the width; old-style playhead via _draw_playhead().
        t_start, t_end = 0.0, total_secs
        state["geom"] = {"x0": plot_x0, "w": plot_w, "y0": plot_y0, "h": plot_h,
                         "total_secs": total_secs}
    else:
        # Scrolling: the playhead sits one bar in from the left; the window is one
        # full segment wide, so ~one bar of history shows behind the line.
        t_start = current_time - sec_per_bar
        t_end   = t_start + total_secs
        state["geom"] = None  # disable old-style moving playhead

    def x_of(t):
        return plot_x0 + ((t - t_start) / (t_end - t_start)) * plot_w

    def y_of(midi):
        return plot_y0 + ((maxp - midi) / span) * (plot_h - row_h)

    # Bar grid lines across the visible window (continuous across segment joins).
    first_bar = int(math.floor(t_start / sec_per_bar))
    last_bar  = int(math.ceil(t_end / sec_per_bar))
    for bar in range(first_bar, last_bar + 1):
        t = bar * sec_per_bar
        x = x_of(t)
        if not (plot_x0 - 1 <= x <= plot_x0 + plot_w + 1):
            continue
        dpg.draw_line((x, plot_y0), (x, plot_y0 + plot_h), color=(55, 55, 62),
                      parent="piano_roll")
        dpg.draw_text((x + 3, plot_y0 + plot_h - 18), str(bar % const.bars), size=13,
                      color=(90, 90, 100), parent="piano_roll")

    # Notes (current segment, plus chained next segment in scrolling mode).
    for evs, offset in segments:
        for name in VOICE_NAMES:
            if name not in evs:
                continue
            color = VOICE_COLORS[name]
            cfg = state["voices"][name]
            for ev in evs[name]:
                vis = max(0.08, ev.duration - cfg.decay)
                n_start = ev.start + offset
                n_end   = min(ev.start + vis, total_secs) + offset
                if n_end < t_start or n_start > t_end:
                    continue
                x0, x1 = x_of(n_start), x_of(n_end)
                y0 = y_of(ev.midi)
                dpg.draw_rectangle((x0, y0), (max(x1, x0 + 2), y0 + row_h * 0.9),
                                   fill=color, color=color, parent="piano_roll")

    # Fixed playhead in scrolling mode: one bar from the left edge.
    if current_time is not None:
        px = x_of(current_time)
        dpg.draw_line((px, plot_y0), (px, plot_y0 + plot_h),
                      color=(255, 70, 70), thickness=2, parent="piano_roll")

    # Legend.
    for i, name in enumerate(VOICE_NAMES):
        on = dpg.get_value(_enabled_tag(name))
        c = VOICE_COLORS[name] if (name in events) else (90, 90, 90)
        lx = plot_x0 + 12 + i * 120
        dpg.draw_rectangle((lx, plot_y0 + 8), (lx + 14, plot_y0 + 22),
                           fill=c, color=c, parent="piano_roll")
        dpg.draw_text((lx + 20, plot_y0 + 7), name if on else f"{name} (off)",
                      size=14, color=(200, 200, 200), parent="piano_roll")


def _draw_playhead() -> None:
    """Moving vertical line at the current playback position."""
    if dpg.does_item_exist("playhead"):
        dpg.delete_item("playhead")
    geom = state["geom"]
    if not state["playing"] or geom is None:
        return
    const = state["const"]
    if const is None:
        return
    total = geom["total_secs"]
    elapsed = state["play_sample"] / const.sample_rate
    if elapsed >= total:
        return
    frac = elapsed / total
    x = geom["x0"] + min(frac, 1.0) * geom["w"]
    dpg.draw_line((x, geom["y0"]), (x, geom["y0"] + geom["h"]),
                  color=(255, 70, 70), thickness=2, tag="playhead", parent="piano_roll")


def frame_update() -> None:
    """Called once per rendered frame to reflect background work + playback."""
    # Generation progress -> bar + status.
    if state["gen_running"]:
        dpg.set_value("gen_progress_bar", state["gen_progress"])
        if state["gen_status"]:
            _status(state["gen_status"])

    # Hand a finished render back to the UI thread.
    if state["gen_result"] is not None:
        const, events, voice_buffers, mixed = state["gen_result"]
        state["gen_result"] = None

        was_playing = state["playing"]
        with _buf_lock:
            old_mixed = state["mixed"]
            if was_playing and old_mixed is not None and mixed is not None:
                # Preserve proportional position so playback continues seamlessly.
                old_frac = state["play_sample"] / len(old_mixed)
                state["mixed"] = mixed
                state["play_sample"] = int(old_frac * len(mixed))
            else:
                state["mixed"] = mixed
                state["play_sample"] = 0

        state["const"] = const
        state["events"] = events
        state["voice_buffers"] = voice_buffers
        state["mixed_seg_len"] = int(const.total_seconds * const.sample_rate)
        # A fresh current segment invalidates any queued endless segment (it was
        # built from the previous params); drop it so preload rebuilds it.
        _invalidate_next()

        # Reopen stream if sample rate changed while playing.
        if was_playing and mixed is not None:
            _ensure_stream(const.sample_rate)

        dpg.set_value("gen_progress_bar", 1.0)
        draw_piano_roll()
        secs = len(mixed) / const.sample_rate if mixed is not None else 0.0
        _status(f"generated {len(events)} voice(s), {secs:.1f}s")

    elif not state["gen_running"] and state["gen_status"] == "cancelled":
        state["gen_status"] = ""
        dpg.set_value("gen_progress_bar", 0.0)
        _status("cancelled")

    # Endless mode: absorb the segment swap signalled by the audio thread. The
    # scrolling redraw at the end of frame_update paints the new segment seamlessly
    # (screen positions are continuous across the swap), so no redraw here.
    state["endless"] = dpg.get_value("session_endless")
    if state["endless_swapped"]:
        state["endless_swapped"] = False
        if state["next_const"] is not None:
            state["const"]         = state["next_const"]
            state["events"]        = state["next_events"]
            state["voice_buffers"] = state["next_voice_buffers"]
            state["next_const"]         = None
            state["next_events"]        = {}
            state["next_voice_buffers"] = {}
            state["next_seg_len"]       = 0
            state["next_tail_merged"]   = False

    # Endless mode: always keep next_buf filled — start preload as soon as the slot is empty.
    if (state["endless"] and state["playing"]
            and not state["endless_gen_running"] and state["next_buf"] is None
            and state["mixed"] is not None and state["const"] is not None):
        _trigger_endless_preload()

    # Endless mode: overlap-add this segment's decay tail into the head of the queued
    # next segment (once), so the ring-out carries over the musical-boundary swap.
    if (state["endless"] and state["next_buf"] is not None
            and not state["next_tail_merged"] and state["mixed"] is not None
            and state["mixed_seg_len"] > 0):
        with _buf_lock:
            cur, seg = state["mixed"], state["mixed_seg_len"]
        if 0 < seg < len(cur):
            tail = cur[seg:]
            nb = state["next_buf"]
            if nb is not None:
                merged = nb.copy()
                m = min(len(tail), len(merged))
                merged[:m] += tail[:m]
                np.clip(merged, -1.0, 1.0, out=merged)
                with _buf_lock:
                    if state["next_buf"] is nb:   # not swapped out while we worked
                        state["next_buf"] = merged
        state["next_tail_merged"] = True

    # Live mode: detect param changes and fire a debounced per-voice re-render.
    state["live_mode"] = dpg.get_value("session_live")

    if state["live_mode"] and not state["gen_running"] and state["mixed"] is not None:
        snap = _param_snapshot()
        baseline = state["param_snapshot"]
        if baseline is None:
            state["param_snapshot"] = snap  # first frame; establish baseline
        elif snap != baseline and state["last_change_t"] == 0.0:
            state["last_change_t"] = time.perf_counter()

        t = state["last_change_t"]
        if t > 0.0 and time.perf_counter() - t >= LIVE_DEBOUNCE_SEC:
            _trigger_live_regen(state["param_snapshot"], snap)

    # Piano roll: scrolling view during playback; static otherwise.
    if state["playing"] and state["const"] is not None and state["mixed"] is not None:
        state["_roll_frame"] = (state["_roll_frame"] + 1) % 2
        if state["_roll_frame"] == 0:
            elapsed = state["play_sample"] / state["const"].sample_rate
            draw_piano_roll(current_time=elapsed)
    else:
        _draw_playhead()  # static-view moving playhead (no-op in scrolling mode)


# --------------------------------------------------------------------------- #
# UI construction.
# --------------------------------------------------------------------------- #
def _tip(tag: str, text: str) -> None:
    with dpg.tooltip(tag):
        dpg.add_text(text, wrap=320)


def _build_session_controls() -> None:
    defaults = SessionConstitution()
    with dpg.collapsing_header(label="Session", default_open=True):
        dpg.add_combo(sorted(NOTE_CLASS), label="key", default_value=defaults.key,
                      tag="session_key", width=140)
        _tip("session_key", SESSION_HELP["key"])
        dpg.add_combo(sorted(SCALES), label="scale", default_value=defaults.scale,
                      tag="session_scale", width=140)
        _tip("session_scale", SESSION_HELP["scale"])
        for name in SESSION_INT_FIELDS:
            if name == "seed":
                with dpg.group(horizontal=True):
                    dpg.add_input_int(label="seed", default_value=defaults.seed,
                                      tag="session_seed", width=140, step=0)
                    _tip("session_seed", SESSION_HELP["seed"])
                    dpg.add_checkbox(label="random", tag="session_seed_random",
                                     default_value=True)
                    _tip("session_seed_random", SESSION_HELP["seed_random"])
                continue
            dpg.add_input_int(label=name, default_value=getattr(defaults, name),
                              tag=f"session_{name}", width=140, step=0)
            _tip(f"session_{name}", SESSION_HELP[name])
        dpg.add_checkbox(label="flat (bypass arrangement)", tag="session_flat")
        _tip("session_flat", SESSION_HELP["flat"])


def _build_voice_controls(name: str) -> None:
    cfg = state["voices"][name]
    with dpg.group(horizontal=True):
        dpg.add_checkbox(tag=_enabled_tag(name), default_value=True)
        _tip(_enabled_tag(name), VOICE_ENABLED_HELP)
        dpg.add_slider_float(tag=f"vol_{name}", default_value=1.0,
                             min_value=0.0, max_value=1.0, width=90, format="%.2f",
                             callback=_on_voice_vol, user_data=name)
        _tip(f"vol_{name}",
             f"Loudness of the {name} voice in the mix. Takes effect instantly "
             "without re-rendering. 0 = silent, 1 = full.")
        with dpg.collapsing_header(label=name):
            for f in dataclasses.fields(VoiceConfig):
                if f.name == "name":
                    continue
                val = getattr(cfg, f.name)
                tag = _voice_tag(name, f.name)
                if isinstance(val, tuple):
                    dpg.add_input_text(label=f.name, default_value=repr(val), tag=tag, width=180)
                elif f.name == "vowel":
                    dpg.add_combo(list(VOWEL_FORMANTS), label=f.name, default_value=val,
                                  tag=tag, width=180)
                elif isinstance(val, bool):
                    dpg.add_checkbox(label=f.name, default_value=val, tag=tag)
                elif isinstance(val, int):
                    dpg.add_input_int(label=f.name, default_value=val, tag=tag, width=180, step=0)
                elif isinstance(val, float):
                    dpg.add_input_double(label=f.name, default_value=val, tag=tag,
                                         width=180, step=0.05, format="%.3f")
                else:
                    dpg.add_input_text(label=f.name, default_value=str(val), tag=tag, width=180)
                _tip(tag, VOICE_HELP.get(f.name, f.name))


def build_ui() -> None:
    with dpg.window(tag="primary"):
        with dpg.group(horizontal=True):
            # left: controls
            with dpg.child_window(width=400, autosize_y=True):
                _build_session_controls()
                for name in VOICE_NAMES:
                    _build_voice_controls(name)
            # right: transport + piano roll
            with dpg.child_window(width=-1, autosize_y=True):
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Generate", callback=on_generate, width=110)
                    dpg.add_button(label="Play", callback=on_play, width=80)
                    dpg.add_button(label="Stop", callback=on_stop, width=80)
                    dpg.add_button(label="Export WAV", callback=on_export, width=110)
                    dpg.add_button(label="Reset Defaults", callback=on_reset_defaults, width=120)
                    dpg.add_checkbox(label="Live", tag="session_live", default_value=True)
                    _tip("session_live",
                         "When on, any parameter change automatically triggers a "
                         "re-render after a short pause. Playback resumes at the same "
                         "point in the song. On/off.")
                    dpg.add_checkbox(label="Endless", tag="session_endless", default_value=True)
                    _tip("session_endless",
                         "When on, a new piece is generated automatically as soon as "
                         "the current one begins playing, so music continues forever "
                         "without repeating. Each new segment uses a fresh random seed. "
                         "On/off.")
                with dpg.group(horizontal=True):
                    dpg.add_progress_bar(tag="gen_progress_bar", width=300, default_value=0.0)
                    dpg.add_button(label="Cancel", callback=on_cancel_gen, width=90)
                    dpg.add_slider_float(label="Volume", tag="master_vol",
                                         default_value=1.0, min_value=0.0, max_value=1.0,
                                         width=200, format="%.2f", callback=_on_master_vol)
                    _tip("master_vol",
                         "Overall output loudness for everything you hear. Takes effect "
                         "instantly. 0 = silent, 1 = full.")
                dpg.add_text("press Generate", tag="status_text")
                with dpg.child_window(width=-1, height=-1, horizontal_scrollbar=True):
                    with dpg.drawlist(width=CANVAS_W, height=CANVAS_H, tag="piano_roll"):
                        pass


def main() -> None:
    dpg.create_context()
    build_ui()
    dpg.create_viewport(title="singing-dolls testbed", width=1500, height=720)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary", True)
    load_settings()
    _sync_volumes_from_widgets()
    draw_piano_roll()
    while dpg.is_dearpygui_running():
        frame_update()
        dpg.render_dearpygui_frame()
    save_settings()
    _close_stream()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
