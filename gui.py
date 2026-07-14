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
import glob
import json
import math
import os
import random
import re
import threading
import time

import numpy as np
import dearpygui.dearpygui as dpg

import generator
import synth
import wav_io
from arrangement import build_arrangement, flat_arrangement
from config import SessionConstitution, VoiceConfig, VOWEL_FORMANTS, default_voices
from generator import GENERATOR_ROLES
from scales import NOTE_CLASS, SCALES

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except Exception:
    _SD_AVAILABLE = False

# Voices are identified by a stable, immutable id (vid) used for widget tags and
# dict keys. The human-facing `name` is just a mutable label, so renaming a voice
# never disturbs tags/state. The default set maps onto vid0..vid3 in build order.
VOICE_PALETTE = (
    (90, 185, 255),   # blue
    (255, 185, 90),   # amber
    (150, 235, 150),  # green
    (235, 120, 200),  # pink
    (255, 140, 120),  # coral
    (140, 200, 255),  # sky
    (210, 210, 120),  # olive
    (180, 140, 240),  # violet
)

CANVAS_W = 1400
CANVAS_H = 560
MARGIN = 40

_HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(_HERE, "gui_settings.json")
VOICES_DIR = os.path.join(_HERE, "voices")


def _vids() -> list:
    """Current voices in display order (list of stable ids)."""
    return list(state["voice_order"])


def _color_for(vid: str) -> tuple:
    return state["voice_colors"].get(vid, (200, 200, 200))

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
    "role": "The kind of part this voice plays, which picks how its notes are written: "
            "'melody' (a roaming lead line), 'harmony' (a calmer inner part), 'drone' (a "
            "sustained low foundation), or 'beat' (vocal drum hits -- pair with the "
            "'percussion' box). Changing this takes full effect on the next Generate.",
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
    "tom_level": "How loud the descending tom drumroll (the fill) sits in the mix. "
                 "Toms share a frequency range with the drone/harmony pads and get "
                 "buried easily -- turn this up to make the roll cut through. 1 = the "
                 "old level.",
}

# --------------------------------------------------------------------------- #
# Voice-editor field grouping. The editor shows only the parameters that matter
# to the selected role; everything is organized into ordered, labeled sections.
# --------------------------------------------------------------------------- #
VOICE_FIELD_GROUPS = [
    ("Core", ["vowel", "octave", "density"]),
    ("Notes", ["walk_steps", "start_degree"]),
    ("Drone", ["degree_pool", "change_bars_min", "change_bars_max"]),
    ("Envelope", ["attack", "decay", "note_frac", "amp"]),
    ("Arrangement", ["chord_lock", "beat_accent", "subdivide", "motif_len",
                     "motif_bars", "energy_response", "answer_parity"]),
    ("Voice character", ["vibrato_hz", "vibrato_semitones", "vibrato_delay",
                         "vibrato_ramp", "tremolo", "scoop_semitones",
                         "scoop_time", "jitter_cents", "shimmer", "breath",
                         "tilt_hz"]),
    ("Percussion", ["beat_steps", "swing", "variation", "ghost_chance",
                    "humanize_ms", "humanize_vel", "fill_chance", "click",
                    "kick_hz", "kick_hz_end", "kick_pitch_tau", "kick_decay",
                    "snare_tone_hz", "snare_noise", "snare_decay", "hat_decay",
                    "openhat_decay", "clap_decay", "rim_decay", "tom_hz",
                    "tom_decay", "tom_level"]),
]

# Which groups each role shows. `percussion` is not a widget -- it is derived
# from the role (see _on_role_changed), keeping the generator (dispatches on
# role) and synth (dispatches on percussion) consistent.
ROLE_FIELD_VISIBILITY = {
    "melody":  {"Core", "Notes", "Envelope", "Arrangement", "Voice character"},
    "harmony": {"Core", "Notes", "Envelope", "Arrangement", "Voice character"},
    "drone":   {"Core", "Drone", "Envelope", "Arrangement", "Voice character"},
    "beat":    {"Core", "Envelope", "Arrangement", "Percussion"},
}
# Field-level overrides where only a subset of a group's fields apply to a role.
ROLE_FIELD_SUBSET = {
    ("drone", "Arrangement"): {"chord_lock", "energy_response", "answer_parity"},
    ("beat", "Core"): {"density"},
    ("beat", "Envelope"): {"amp", "decay"},
    ("beat", "Arrangement"): {"energy_response", "answer_parity", "beat_accent"},
}


def _visible_fields(role):
    """Ordered [(group_label, [field_names])] the editor should show for `role`."""
    visible = ROLE_FIELD_VISIBILITY.get(role, ROLE_FIELD_VISIBILITY["melody"])
    out = []
    for label, fields in VOICE_FIELD_GROUPS:
        if label not in visible:
            continue
        subset = ROLE_FIELD_SUBSET.get((role, label))
        shown = [f for f in fields if subset is None or f in subset]
        if shown:
            out.append((label, shown))
    return out


# Mini-piano: keyboard key -> (display name, scale semitone above C). Pitched
# roles play these as C-G; the beat role maps the five keys to drum hits instead.
_PIANO_KEYS = [("a", "C", 0), ("s", "D", 2), ("d", "E", 4), ("f", "F", 5),
               ("g", "G", 7)]
_PIANO_DRUMS = ["kick", "snare", "hat", "openhat", "clap"]
# Which decay knob shapes each drum's tail (for the envelope visualization).
_DRUM_DECAY_FIELD = ["kick_decay", "snare_decay", "hat_decay", "openhat_decay",
                     "clap_decay"]

# Parameter-shape mini-graphs shown in the editor so a layman can *see* how the
# knobs change the sound: an amplitude envelope and a pitch (scoop+vibrato) curve.
VIZ_W, VIZ_H, VIZ_PAD = 440, 90, 12

# --------------------------------------------------------------------------- #
# Live state.
# --------------------------------------------------------------------------- #

# Guards state["mixed"] and state["play_sample"] during buffer swaps.
# The audio callback runs on a PortAudio thread, so we need this to prevent
# a torn read while we swap in a freshly rendered buffer.
_buf_lock = threading.Lock()

state = {
    # --- voice collection (stable-id keyed; see VOICE_PALETTE note above) ---
    "voices": {},              # {vid: VoiceConfig}
    "voice_order": [],         # [vid, ...] display order
    "voice_colors": {},        # {vid: (r,g,b)}
    "voice_enabled": {},       # {vid: bool}   mirror of the enabled checkboxes
    "_next_vid": 0,            # monotonic counter for minting new vids
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
    "voice_vol": {},           # {vid: gain} per-voice gains applied at mix time
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
    # --- voice editor / mini-piano ---
    "editing_vid": None,       # vid whose editor window currently owns the piano keys
    "piano_oct": {},           # {vid: octave offset from C4} for the mini-piano
    "piano_drum": {},          # {vid: last-sampled drum index} -> which envelope to draw
}


# --------------------------------------------------------------------------- #
# Voice collection management (add / remove / (de)serialize).
# --------------------------------------------------------------------------- #
def _mint_vid() -> str:
    vid = f"vid{state['_next_vid']}"
    state["_next_vid"] += 1
    return vid


def _register_voice(cfg: VoiceConfig, vid: str | None = None, *,
                    enabled: bool = True, vol: float = 1.0,
                    color: tuple | None = None) -> str:
    """Insert a voice into the collection and return its (possibly new) vid."""
    if vid is None:
        vid = _mint_vid()
    state["voices"][vid] = cfg
    state["voice_order"].append(vid)
    state["voice_enabled"][vid] = enabled
    state["voice_vol"][vid] = vol
    state["voice_colors"][vid] = color or VOICE_PALETTE[
        (len(state["voice_order"]) - 1) % len(VOICE_PALETTE)]
    return vid


def _forget_voice(vid: str) -> None:
    """Remove a voice and all of its per-vid state."""
    for d in ("voices", "voice_enabled", "voice_vol", "voice_colors"):
        state[d].pop(vid, None)
    if vid in state["voice_order"]:
        state["voice_order"].remove(vid)
    state["events"].pop(vid, None)
    state["voice_buffers"].pop(vid, None)


def _reset_voice_collection() -> None:
    """Rebuild the collection from the four factory-default dolls."""
    state["voices"], state["voice_order"] = {}, []
    state["voice_enabled"], state["voice_vol"], state["voice_colors"] = {}, {}, {}
    state["_next_vid"] = 0
    for cfg in default_voices().values():
        _register_voice(cfg)


def _unique_name(name: str) -> str:
    """Disambiguate against current voice names (the generator's per-voice PRNG is
    keyed on name, so identical names would render identically)."""
    existing = {state["voices"][v].name for v in state["voice_order"]}
    if name not in existing:
        return name
    i = 2
    while f"{name} {i}" in existing:
        i += 1
    return f"{name} {i}"


def _new_blank_voice() -> VoiceConfig:
    """A sensible fresh sung voice for the 'Add Voice' button."""
    n = len(state["voice_order"]) + 1
    return VoiceConfig(name=_unique_name(f"voice{n}"), vowel="oo", octave=0, density=0.4)


# --------------------------------------------------------------------------- #
# Voice (de)serialization + on-disk library (voices/*.json).
# --------------------------------------------------------------------------- #
def _voice_to_dict(cfg: VoiceConfig) -> dict:
    """Plain-JSON dict of a VoiceConfig (tuples become lists via asdict)."""
    return dataclasses.asdict(cfg)


def _voice_from_dict(d: dict) -> VoiceConfig:
    """Rebuild a VoiceConfig, coercing each field to its declared type."""
    cfg = VoiceConfig(
        name=str(d.get("name", "voice")),
        vowel=str(d.get("vowel", "oo")),
        octave=int(d.get("octave", 0)),
        density=float(d.get("density", 0.4)),
    )
    for f in dataclasses.fields(VoiceConfig):
        if f.name in ("name", "vowel", "octave", "density") or f.name not in d:
            continue
        cur, val = getattr(cfg, f.name), d[f.name]
        if isinstance(cur, tuple):
            setattr(cfg, f.name, tuple(val))
        elif isinstance(cur, bool):
            setattr(cfg, f.name, bool(val))
        elif isinstance(cur, int):
            setattr(cfg, f.name, int(val))
        elif isinstance(cur, float):
            setattr(cfg, f.name, float(val))
        else:
            setattr(cfg, f.name, val)
    return cfg


def _library_path(name: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", name.strip()) or "voice"
    return os.path.join(VOICES_DIR, f"{slug}.json")


def library_voices() -> list:
    """Names (filename stems) of voices saved in the library, sorted."""
    if not os.path.isdir(VOICES_DIR):
        return []
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(VOICES_DIR, "*.json")))


# Populate the default collection at import time so the UI can build from it.
_reset_voice_collection()


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
def _voice_tag(vid: str, field: str) -> str:
    return f"v_{vid}_{field}"


def _enabled_tag(vid: str) -> str:
    return f"v_{vid}_enabled"


def _vol_tag(vid: str) -> str:
    return f"vol_{vid}"


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
    """Push every voice widget value back into its VoiceConfig object.

    `name` is handled live by _on_rename (its widget is a plain label editor, not
    a config knob), so it is skipped here along with any voice whose widgets have
    been torn down mid-rebuild."""
    for vid in _vids():
        cfg = state["voices"][vid]
        for f in dataclasses.fields(VoiceConfig):
            if f.name == "name":
                continue
            tag = _voice_tag(vid, f.name)
            if not dpg.does_item_exist(tag):
                continue
            raw = dpg.get_value(tag)
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
        for i, (vid, cfg) in enumerate(enabled):
            if state["gen_cancel"]:
                state["gen_status"] = "cancelled"
                state["gen_running"] = False
                return
            state["gen_status"] = f"rendering {cfg.name}..."
            evs = generator.generate_voice(const, cfg, arr)
            events[vid] = evs
            buf = synth.render_events(evs, const, cfg)
            voice_buffers[vid] = buf   # store unscaled; per-voice gain applied at mix
            buffers.append(buf * state["voice_vol"].get(vid, 1.0))
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
    for vid in _vids():
        if not dpg.get_value(_enabled_tag(vid)):
            continue
        cfg = state["voices"][vid]
        if flat:
            _flatten_voice(cfg)
        enabled.append((vid, cfg))

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
    buffers = [vb[vid] * state["voice_vol"].get(vid, 1.0)
               for vid in _vids() if vid in vb]
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
        buffers = [nvb[vid] * state["voice_vol"].get(vid, 1.0)
                   for vid in _vids() if vid in nvb]
        state["next_buf"] = synth.mix(buffers) if buffers else None
        state["next_tail_merged"] = False


def _sync_volumes_from_widgets() -> None:
    """Pull slider values into state (set_value does not fire callbacks)."""
    if dpg.does_item_exist("master_vol"):
        state["master_vol"] = float(dpg.get_value("master_vol"))
    for vid in _vids():
        tag = _vol_tag(vid)
        if dpg.does_item_exist(tag):
            state["voice_vol"][vid] = float(dpg.get_value(tag))


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
# Persistence: remember the session + the whole (dynamic) voice set.
# --------------------------------------------------------------------------- #
def _global_tags():
    """Session-level widget tags (everything that isn't a per-voice control)."""
    tags = ["session_key", "session_scale", "session_seed_random", "session_flat",
            "session_live", "session_endless"]
    tags += [f"session_{n}" for n in SESSION_INT_FIELDS]
    return tags


def _voice_snapshot_tags():
    """Per-voice widget tags watched for live re-render. `name` and the volume
    faders are deliberately excluded: renaming/refading must not force a render."""
    tags = []
    for vid in _vids():
        tags.append(_enabled_tag(vid))
        for f in dataclasses.fields(VoiceConfig):
            if f.name != "name":
                tags.append(_voice_tag(vid, f.name))
    return tags


def _sync_all_from_widgets() -> None:
    """Pull every voice widget value (fields, name, enabled, volume) into state,
    so a save reflects the current UI even outside live mode."""
    _apply_voice_widgets()
    _sync_volumes_from_widgets()
    for vid in _vids():
        cfg = state["voices"].get(vid)
        if cfg is None:
            continue
        ntag, etag = _voice_tag(vid, "name"), _enabled_tag(vid)
        if dpg.does_item_exist(ntag):
            cfg.name = str(dpg.get_value(ntag))
        if dpg.does_item_exist(etag):
            state["voice_enabled"][vid] = bool(dpg.get_value(etag))


def save_settings() -> None:
    _sync_all_from_widgets()
    data = {tag: dpg.get_value(tag) for tag in _global_tags() if dpg.does_item_exist(tag)}
    if dpg.does_item_exist("master_vol"):
        data["master_vol"] = dpg.get_value("master_vol")
    data["voices"] = [
        {
            "enabled": state["voice_enabled"].get(vid, True),
            "vol": state["voice_vol"].get(vid, 1.0),
            "color": list(_color_for(vid)),
            "config": _voice_to_dict(state["voices"][vid]),
        }
        for vid in _vids()
    ]
    try:
        with open(SETTINGS_PATH, "w") as fh:
            json.dump(data, fh, indent=2)
    except OSError as exc:
        print(f"[gui] could not save settings: {exc}")


def load_settings() -> None:
    """Restore session widgets and rebuild the voice set from disk (if present).
    Rebuilds the voice panels, so must run after build_ui()."""
    if not os.path.exists(SETTINGS_PATH):
        return
    try:
        with open(SETTINGS_PATH) as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"[gui] could not load settings: {exc}")
        return

    for tag in _global_tags() + ["master_vol"]:
        if tag in data and dpg.does_item_exist(tag):
            try:
                dpg.set_value(tag, data[tag])
            except Exception:
                pass

    voices = data.get("voices")
    if isinstance(voices, list) and voices:
        _close_editors()                 # configs are replaced wholesale below
        state["voices"], state["voice_order"] = {}, []
        state["voice_enabled"], state["voice_vol"], state["voice_colors"] = {}, {}, {}
        state["_next_vid"] = 0
        for entry in voices:
            cfg = _voice_from_dict(entry.get("config", {}))
            color = tuple(entry["color"]) if entry.get("color") else None
            _register_voice(cfg, enabled=bool(entry.get("enabled", True)),
                            vol=float(entry.get("vol", 1.0)), color=color)
        _rebuild_voice_panels()


def _default_settings() -> dict:
    """Factory-default values for the session-level widgets."""
    const = SessionConstitution()
    out = {
        "session_key": const.key,
        "session_scale": const.scale,
        "session_seed_random": True,
        "session_flat": False,
        "session_live": True,
        "session_endless": True,
        "master_vol": 1.0,
    }
    for n in SESSION_INT_FIELDS:
        out[f"session_{n}"] = getattr(const, n)
    return out


def on_reset_defaults() -> None:
    _close_editors()                     # configs are replaced wholesale below
    _reset_voice_collection()
    for tag, val in _default_settings().items():
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, val)
    _rebuild_voice_panels()
    _sync_volumes_from_widgets()
    state["events"], state["voice_buffers"] = {}, {}
    with _buf_lock:
        state["mixed"] = None
        state["play_sample"] = 0
    _invalidate_next()
    state["param_snapshot"] = _param_snapshot()
    draw_piano_roll()
    _status("reset to defaults")


# --------------------------------------------------------------------------- #
# Live mode helpers.
# --------------------------------------------------------------------------- #
def _param_snapshot() -> dict:
    """Snapshot all controllable widget values for change detection."""
    return {tag: dpg.get_value(tag)
            for tag in _global_tags() + _voice_snapshot_tags()
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
        for vid in _vids():
            if tag.startswith(f"v_{vid}_"):
                matched = vid
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
        for idx, vid in enumerate(names):
            state["gen_progress"] = (idx + 1) / (n_voices + 1)
            cfg = state["voices"].get(vid)
            state["gen_status"] = f"updating {cfg.name if cfg else vid}..."
            if cfg is None or not dpg.get_value(_enabled_tag(vid)):
                voice_buffers.pop(vid, None)
                continue
            evs = events.get(vid, [])
            if evs:
                voice_buffers[vid] = synth.render_events(evs, const, cfg)

        buffers = [voice_buffers[vid] * state["voice_vol"].get(vid, 1.0)
                   for vid in _vids() if vid in voice_buffers]
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
        for vid, cfg in enabled:
            evs = generator.generate_voice(const, cfg, arr)
            events[vid] = evs
            buf = synth.render_events(evs, const, cfg)
            voice_buffers[vid] = buf   # store unscaled; per-voice gain applied at mix
            buffers.append(buf * state["voice_vol"].get(vid, 1.0))
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
    enabled = [(vid, state["voices"][vid]) for vid in _vids()
               if dpg.get_value(_enabled_tag(vid))]
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
        for vid in _vids():
            if vid not in evs:
                continue
            color = _color_for(vid)
            cfg = state["voices"][vid]
            for ev in evs[vid]:
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

    # Legend (wraps to a second row if there are many voices).
    for i, vid in enumerate(_vids()):
        cfg = state["voices"][vid]
        on = state["voice_enabled"].get(vid, True)
        c = _color_for(vid) if (vid in events) else (90, 90, 90)
        lx = plot_x0 + 12 + (i % 6) * 120
        ly = plot_y0 + 8 + (i // 6) * 22
        dpg.draw_rectangle((lx, ly), (lx + 14, ly + 14),
                           fill=c, color=c, parent="piano_roll")
        dpg.draw_text((lx + 20, ly - 1), cfg.name if on else f"{cfg.name} (off)",
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


# --- per-voice control callbacks --------------------------------------------
def _on_enabled(sender, app_data, user_data) -> None:
    state["voice_enabled"][user_data] = bool(app_data)
    if not state["playing"]:
        draw_piano_roll()  # refresh the legend's on/off label


def _on_rename(sender, app_data, user_data) -> None:
    vid = user_data
    cfg = state["voices"].get(vid)
    if cfg is None:
        return
    cfg.name = str(app_data)
    win = _edit_win(vid)
    if dpg.does_item_exist(win):
        dpg.configure_item(win, label=f"Edit: {cfg.name or '(unnamed)'}")
    if not state["playing"]:
        draw_piano_roll()  # rename shows in the legend; no re-render needed


def on_add_voice() -> None:
    _apply_voice_widgets()               # preserve edits in the other panels
    vid = _register_voice(_new_blank_voice())
    _rebuild_voice_panels()
    state["param_snapshot"] = _param_snapshot()
    if not state["playing"]:
        draw_piano_roll()
    _status(f"added voice '{state['voices'][vid].name}' -- press Generate to hear it")


def on_delete_voice(sender, app_data, user_data) -> None:
    vid = user_data
    _apply_voice_widgets()               # preserve edits in the surviving panels
    name = state["voices"].get(vid).name if vid in state["voices"] else vid
    if dpg.does_item_exist(_edit_win(vid)):
        dpg.delete_item(_edit_win(vid))  # close its editor before the voice is gone
    if state["editing_vid"] == vid:
        state["editing_vid"] = None
    _forget_voice(vid)
    _rebuild_voice_panels()
    _remix_current()                     # drop its audio from the live mix now
    _invalidate_next()
    state["param_snapshot"] = _param_snapshot()
    draw_piano_roll()
    _status(f"deleted voice '{name}'")


def on_save_voice(sender, app_data, user_data) -> None:
    vid = user_data
    _apply_voice_widgets()
    cfg = state["voices"].get(vid)
    if cfg is None:
        return
    ntag = _voice_tag(vid, "name")
    if dpg.does_item_exist(ntag):
        cfg.name = str(dpg.get_value(ntag))
    try:
        os.makedirs(VOICES_DIR, exist_ok=True)
        path = _library_path(cfg.name)
        with open(path, "w") as fh:
            json.dump(_voice_to_dict(cfg), fh, indent=2)
    except OSError as exc:
        _status(f"could not save voice: {exc}")
        return
    _refresh_library_combo()
    _status(f"saved voice to {os.path.relpath(path, _HERE)}")


def on_add_from_library() -> None:
    name = dpg.get_value("library_combo")
    if not name:
        _status("pick a saved voice from the list first")
        return
    path = _library_path(name)
    try:
        with open(path) as fh:
            cfg = _voice_from_dict(json.load(fh))
    except (OSError, ValueError) as exc:
        _status(f"could not load voice: {exc}")
        return
    _apply_voice_widgets()
    cfg.name = _unique_name(cfg.name)
    vid = _register_voice(cfg)
    _rebuild_voice_panels()
    state["param_snapshot"] = _param_snapshot()
    if not state["playing"]:
        draw_piano_roll()
    _status(f"added '{cfg.name}' from library -- press Generate to hear it")


def _refresh_library_combo() -> None:
    if dpg.does_item_exist("library_combo"):
        dpg.configure_item("library_combo", items=library_voices())


# --- per-voice panel builder ------------------------------------------------
def _add_field_widget(vid: str, fname: str, val) -> None:
    """Add the editing widget for one VoiceConfig field into the current container.
    Shared by the voice row's role combo and the editor window's grouped fields."""
    tag = _voice_tag(vid, fname)
    cb = _on_voice_param_edit          # refresh the shape graphs on any edit
    if isinstance(val, tuple):
        dpg.add_input_text(label=fname, default_value=repr(val), tag=tag, width=180,
                           callback=cb, user_data=vid)
    elif fname == "role":
        dpg.add_combo(list(GENERATOR_ROLES), label=fname, default_value=val,
                      tag=tag, width=180, callback=_on_role_changed, user_data=vid)
    elif fname == "vowel":
        dpg.add_combo(list(VOWEL_FORMANTS), label=fname, default_value=val,
                      tag=tag, width=180, callback=cb, user_data=vid)
    elif isinstance(val, bool):
        dpg.add_checkbox(label=fname, default_value=val, tag=tag,
                         callback=cb, user_data=vid)
    elif isinstance(val, int):
        dpg.add_input_int(label=fname, default_value=val, tag=tag, width=180, step=0,
                          callback=cb, user_data=vid)
    elif isinstance(val, float):
        dpg.add_input_double(label=fname, default_value=val, tag=tag,
                             width=180, step=0.05, format="%.3f",
                             callback=cb, user_data=vid)
    else:
        dpg.add_input_text(label=fname, default_value=str(val), tag=tag, width=180,
                           callback=cb, user_data=vid)
    _tip(tag, VOICE_HELP.get(fname, fname))


def _build_voice_panel(vid: str) -> None:
    """Build one voice's row as a child of the 'voices_panel' group. Detailed
    parameter editing lives in a separate popup opened by the 'Edit' button."""
    cfg = state["voices"][vid]
    with dpg.group(parent="voices_panel"):
        with dpg.group(horizontal=True):
            dpg.add_checkbox(tag=_enabled_tag(vid),
                             default_value=state["voice_enabled"].get(vid, True),
                             callback=_on_enabled, user_data=vid)
            _tip(_enabled_tag(vid), VOICE_ENABLED_HELP)
            dpg.add_slider_float(tag=_vol_tag(vid),
                                 default_value=state["voice_vol"].get(vid, 1.0),
                                 min_value=0.0, max_value=1.0, width=80, format="%.2f",
                                 callback=_on_voice_vol, user_data=vid)
            _tip(_vol_tag(vid),
                 "Loudness of this voice in the mix. Takes effect instantly "
                 "without re-rendering. 0 = silent, 1 = full.")
            dpg.add_input_text(tag=_voice_tag(vid, "name"), default_value=cfg.name,
                               width=100, callback=_on_rename, user_data=vid)
            _tip(_voice_tag(vid, "name"),
                 "This voice's name -- edit freely; it only relabels the voice "
                 "(shown in the piano-roll legend) and doesn't change the sound.")
            dpg.add_button(label="Edit", width=42, callback=on_edit_voice, user_data=vid)
            _tip(dpg.last_item(),
                 "Open this voice's editor: role-specific parameters plus a mini "
                 "piano to audition how it sounds.")
            dpg.add_button(label="Save", width=42, callback=on_save_voice, user_data=vid)
            _tip(dpg.last_item(), "Save this voice to the reusable library (voices/).")
            dpg.add_button(label="Del", width=38, callback=on_delete_voice, user_data=vid)
            _tip(dpg.last_item(), "Remove this voice from the session.")
        dpg.add_separator()


# --- voice editor window ----------------------------------------------------
def _edit_win(vid: str) -> str:
    return f"edit_win_{vid}"


def _build_editor_fields(vid: str) -> None:
    """(Re)build the role-filtered parameter widgets inside the editor's field
    container. Called on open and whenever the role changes."""
    cfg = state["voices"][vid]
    dpg.push_container_stack(f"edit_fields_{vid}")
    try:
        for label, fields in _visible_fields(cfg.role):
            dpg.add_text(label)
            for fname in fields:
                _add_field_widget(vid, fname, getattr(cfg, fname))
            dpg.add_separator()
    finally:
        dpg.pop_container_stack()


def on_edit_voice(sender, app_data, user_data) -> None:
    """Open (or focus) the popup editor for a voice."""
    vid = user_data
    win = _edit_win(vid)
    if dpg.does_item_exist(win):
        dpg.focus_item(win)
        state["editing_vid"] = vid
        return
    cfg = state["voices"][vid]
    state["piano_oct"].setdefault(vid, 0)
    with dpg.window(label=f"Edit: {cfg.name}", tag=win, width=470, height=680,
                    pos=(430, 30), on_close=_on_editor_close, user_data=vid):
        _add_field_widget(vid, "role", cfg.role)
        dpg.add_group(tag=f"viz_box_{vid}")   # amplitude + pitch shape graphs
        _build_voice_viz(vid)
        dpg.add_separator()
        # Fields scroll in their own box that fills the window down to ~140px from
        # the bottom -- so resizing taller gives more editing room while the piano
        # block stays pinned to the bottom. (Negative height = reserve that margin.)
        dpg.add_child_window(tag=f"edit_fields_{vid}", width=-1, height=-140)
        _build_editor_fields(vid)
        dpg.add_separator()
        _build_piano(vid)
    state["editing_vid"] = vid


def _on_editor_close(sender, app_data, user_data) -> None:
    """Capture final edits into the config before the widgets are torn down."""
    vid = user_data
    _apply_voice_widgets()
    if state["editing_vid"] == vid:
        state["editing_vid"] = None
    if dpg.does_item_exist(_edit_win(vid)):
        dpg.delete_item(_edit_win(vid))


def _on_role_changed(sender, app_data, user_data) -> None:
    """Role picked in the editor: re-derive percussion and rebuild the visible
    parameter set for the new role (hidden fields keep their config values)."""
    vid = user_data
    _apply_voice_widgets()
    cfg = state["voices"][vid]
    cfg.role = app_data
    cfg.percussion = (app_data == "beat")
    fields_tag = f"edit_fields_{vid}"
    if dpg.does_item_exist(fields_tag):
        dpg.delete_item(fields_tag, children_only=True)
        _build_editor_fields(vid)
    viz_tag = f"viz_box_{vid}"
    if dpg.does_item_exist(viz_tag):        # pitch graph appears only for pitched roles
        dpg.delete_item(viz_tag, children_only=True)
        _build_voice_viz(vid)
    _update_piano_label(vid)


# --- mini piano + independent note preview ----------------------------------
def _octave_label(vid: str) -> str:
    return f"C{4 + state['piano_oct'].get(vid, 0)}"


def _update_piano_label(vid: str) -> None:
    tag = f"piano_octlbl_{vid}"
    if dpg.does_item_exist(tag):
        dpg.set_value(tag, _octave_label(vid))


# --- parameter-shape visualizations -----------------------------------------
def _viz_get(vid: str, fname: str, default: float) -> float:
    """Current numeric value of a field -- from its live widget if the editor is
    open, else the stored config -- so the graphs track edits before Generate."""
    tag = _voice_tag(vid, fname)
    if dpg.does_item_exist(tag):
        try:
            return float(dpg.get_value(tag))
        except (TypeError, ValueError):
            return default
    return float(getattr(state["voices"].get(vid, None), fname, default))


def _build_voice_viz(vid: str) -> None:
    """(Re)build the graph area for a voice: an amplitude envelope always, plus a
    pitch curve for pitched roles (a drone/melody/harmony -- not the beatbox)."""
    parent = f"viz_box_{vid}"
    beat = (state["voices"][vid].role == "beat")
    dpg.push_container_stack(parent)
    try:
        lbl = "Drum-hit shape (loudness over time)" if beat else \
              "Note shape -- loudness over time (swell in / hold / fade out)"
        dpg.add_text(lbl, tag=f"viz_envlbl_{vid}")
        with dpg.drawlist(width=VIZ_W, height=VIZ_H, tag=f"viz_env_{vid}"):
            pass
        if not beat:
            dpg.add_text("Pitch -- scoop up into the note, then vibrato wobble")
            with dpg.drawlist(width=VIZ_W, height=VIZ_H, tag=f"viz_pitch_{vid}"):
                pass
    finally:
        dpg.pop_container_stack()
    _redraw_voice_viz(vid)


def _redraw_voice_viz(vid: str) -> None:
    if dpg.does_item_exist(f"viz_env_{vid}"):
        _draw_envelope(vid)
    if dpg.does_item_exist(f"viz_pitch_{vid}"):
        _draw_pitch(vid)


_VIZ_AXIS = (90, 90, 90)
_VIZ_ENV_COLOR = (120, 200, 255)
_VIZ_PITCH_COLOR = (255, 185, 90)


def _draw_envelope(vid: str) -> None:
    """Loudness of a single note over time. Shape only (a representative slot);
    the point is relative feel -- taller = louder, longer tail = longer decay."""
    tag = f"viz_env_{vid}"
    dpg.delete_item(tag, children_only=True)
    beat = (state["voices"][vid].role == "beat")
    x0, x1, ybot, ytop = VIZ_PAD, VIZ_W - VIZ_PAD, VIZ_H - VIZ_PAD, VIZ_PAD
    if beat:
        idx = state["piano_drum"].get(vid, 0)
        attack, hold = 0.002, 0.0
        decay = max(_viz_get(vid, _DRUM_DECAY_FIELD[idx], 0.15), 1e-3)
        dpg.set_value(f"viz_envlbl_{vid}",
                      f"Drum-hit shape: {_PIANO_DRUMS[idx]} (loudness over time)")
    else:
        attack = max(_viz_get(vid, "attack", 0.02), 1e-4)
        hold = max(_viz_get(vid, "note_frac", 0.9), 0.0) * 1.0  # 1s representative slot
        decay = max(_viz_get(vid, "decay", 2.5), 1e-3)
    amp = max(0.0, min(_viz_get(vid, "amp", 0.9), 1.0))
    total = attack + hold + decay
    n = 140
    pts = []
    for i in range(n):
        t = total * i / (n - 1)
        if t < attack:
            e = 1.0 - math.exp(-3.0 * t / attack)
        elif t < attack + hold:
            e = 1.0
        else:
            e = math.exp(-3.0 * (t - attack - hold) / decay)
        x = x0 + (t / total) * (x1 - x0)
        y = ybot - (e * amp) * (ybot - ytop)
        pts.append([x, y])
    dpg.draw_line([x0, ybot], [x1, ybot], color=_VIZ_AXIS, parent=tag)
    dpg.draw_line([x0, ybot], [x0, ytop], color=_VIZ_AXIS, parent=tag)
    dpg.draw_polyline(pts, color=_VIZ_ENV_COLOR, thickness=2, parent=tag)


def _draw_pitch(vid: str) -> None:
    """Pitch relative to the target note: a scoop up at the onset, then vibrato
    wobble that eases in. The flat grey line is 'in tune' (the target pitch)."""
    tag = f"viz_pitch_{vid}"
    dpg.delete_item(tag, children_only=True)
    scoop = _viz_get(vid, "scoop_semitones", 0.4)
    stime = max(_viz_get(vid, "scoop_time", 0.07), 1e-3)
    vhz = _viz_get(vid, "vibrato_hz", 5.5)
    vsemi = _viz_get(vid, "vibrato_semitones", 0.35)
    vdelay = _viz_get(vid, "vibrato_delay", 0.35)
    vramp = max(_viz_get(vid, "vibrato_ramp", 0.6), 1e-3)
    x0, x1, ymid = VIZ_PAD, VIZ_W - VIZ_PAD, VIZ_H / 2.0
    span = max(scoop, vsemi, 0.5) * 1.15
    T, n = 1.4, 240
    pts = []
    for i in range(n):
        t = T * i / (n - 1)
        s = -scoop * math.exp(-t / stime)
        if t > vdelay:
            dep = vsemi * min((t - vdelay) / vramp, 1.0)
            v = dep * math.sin(2.0 * math.pi * vhz * (t - vdelay))
        else:
            v = 0.0
        x = x0 + (t / T) * (x1 - x0)
        y = ymid - ((s + v) / span) * (ymid - VIZ_PAD)
        pts.append([x, y])
    dpg.draw_line([x0, ymid], [x1, ymid], color=_VIZ_AXIS, parent=tag)
    dpg.draw_polyline(pts, color=_VIZ_PITCH_COLOR, thickness=2, parent=tag)


def _on_voice_param_edit(sender, app_data, user_data) -> None:
    """Any editor knob changed: refresh this voice's shape graphs live."""
    _redraw_voice_viz(user_data)


def _build_piano(vid: str) -> None:
    cfg = state["voices"][vid]
    beat = (cfg.role == "beat")
    dpg.add_text("Sample this voice -- keys: a s d f g")
    with dpg.group(horizontal=True):
        for i, (key, name, _semi) in enumerate(_PIANO_KEYS):
            cap = _PIANO_DRUMS[i] if beat else name
            dpg.add_button(label=f"{cap}\n({key})", width=54, height=44,
                           callback=_on_piano_button, user_data=(vid, i))
    with dpg.group(horizontal=True):
        dpg.add_button(label=" - ", callback=_on_octave, user_data=(vid, -1))
        _tip(dpg.last_item(), "Drop the piano an octave.")
        dpg.add_text(_octave_label(vid), tag=f"piano_octlbl_{vid}")
        dpg.add_button(label=" + ", callback=_on_octave, user_data=(vid, 1))
        _tip(dpg.last_item(), "Raise the piano an octave.")
        dpg.add_text("  (octave -- ignored for beat voices)")


def _on_octave(sender, app_data, user_data) -> None:
    vid, delta = user_data
    o = max(-3, min(3, state["piano_oct"].get(vid, 0) + delta))
    state["piano_oct"][vid] = o
    _update_piano_label(vid)


def _on_piano_button(sender, app_data, user_data) -> None:
    vid, idx = user_data
    _preview_note(vid, idx)


def _preview_note(vid: str, idx: int) -> None:
    """Render a single note (or drum hit) for this voice with its current, unsaved
    edits and play it on an independent channel over whatever else is playing."""
    if not _SD_AVAILABLE:
        _status("preview unavailable (sounddevice not installed)")
        return
    if vid not in state["voices"]:
        return
    try:
        _apply_voice_widgets()
        const = _read_const()
    except (ValueError, SyntaxError) as exc:
        _status(f"invalid parameter: {exc}")
        return
    cfg = state["voices"][vid]
    if cfg.percussion:
        state["piano_drum"][vid] = idx      # graph the drum we're auditioning
        _redraw_voice_viz(vid)
        ev = generator.NoteEvent(midi=60, start=0.0, duration=0.5,
                                 voice=cfg.name, sound=_PIANO_DRUMS[idx])
    else:
        midi = 60 + 12 * state["piano_oct"].get(vid, 0) + _PIANO_KEYS[idx][2]
        dur = max(1.0, cfg.attack + 0.6 + cfg.decay)
        ev = generator.NoteEvent(midi=midi, start=0.0, duration=dur,
                                 voice=cfg.name, velocity=1.0)
    try:
        # render_events pads to the whole session length; the note lives at the
        # front, so trim to it (plus a hair) instead of playing minutes of silence.
        buf = synth.render_events([ev], const, cfg)
        n = min(len(buf), int(ev.duration * const.sample_rate) + 1)
        sd.play(np.asarray(buf[:n], dtype=np.float32), const.sample_rate)
    except Exception as exc:
        _status(f"preview error: {exc}")


def _active_editor_vid():
    """The vid of the editor that should receive piano keystrokes, or None. Keys
    only act while an editor window is open (so typing elsewhere is unaffected)."""
    vid = state["editing_vid"]
    if vid is None or vid not in state["voices"]:
        return None
    if not dpg.does_item_exist(_edit_win(vid)):
        state["editing_vid"] = None
        return None
    return vid


_PIANO_KEY_INDEX = {key: i for i, (key, _n, _s) in enumerate(_PIANO_KEYS)}

# Item types that consume typing -- while one of these is focused the piano keys
# must not fire (e.g. a '-' typed into walk_steps shouldn't shift the octave).
_TEXT_INPUT_TYPES = ("InputText", "InputInt", "InputDouble", "InputFloat", "Combo")


def _text_input_focused() -> bool:
    item = dpg.get_focused_item()
    if not item:
        return False
    try:
        t = dpg.get_item_type(item)
    except Exception:
        return False
    return any(k in t for k in _TEXT_INPUT_TYPES)


def _on_piano_key(sender, app_data, user_data) -> None:
    if _text_input_focused():
        return
    vid = _active_editor_vid()
    if vid is not None:
        _preview_note(vid, _PIANO_KEY_INDEX[user_data])


def _on_piano_octkey(sender, app_data, user_data) -> None:
    if _text_input_focused():
        return
    vid = _active_editor_vid()
    if vid is not None:
        _on_octave(None, None, (vid, user_data))


def _register_piano_keys() -> None:
    """Global key handlers driving the mini-piano of the focused editor window."""
    with dpg.handler_registry():
        for key, _name, _semi in _PIANO_KEYS:
            k = getattr(dpg, f"mvKey_{key.upper()}", None)
            if k is not None:
                dpg.add_key_press_handler(k, callback=_on_piano_key, user_data=key)
        for name in ("mvKey_Minus", "mvKey_Subtract"):
            k = getattr(dpg, name, None)
            if k is not None:
                dpg.add_key_press_handler(k, callback=_on_piano_octkey, user_data=-1)
        for name in ("mvKey_Plus", "mvKey_Add"):
            k = getattr(dpg, name, None)
            if k is not None:
                dpg.add_key_press_handler(k, callback=_on_piano_octkey, user_data=1)


def _close_editors(keep=None) -> None:
    """Close editor windows. With keep=set(vids), only those whose voice is absent
    are closed (used on add/delete); with keep=None, close every editor (used when
    the whole collection is replaced by reset/load, which reuses vids for new
    configs, so open editors would show stale widgets)."""
    for alias in list(dpg.get_aliases()):
        if not alias.startswith("edit_win_"):
            continue
        vid = alias[len("edit_win_"):]
        if (keep is None or vid not in keep) and dpg.does_item_exist(alias):
            dpg.delete_item(alias)
    if keep is None or state["editing_vid"] not in keep:
        state["editing_vid"] = None


def _rebuild_voice_panels() -> None:
    """Tear down and rebuild every voice panel from state (after add/delete/load).
    State is the source of truth; widgets are recreated from it."""
    if not dpg.does_item_exist("voices_panel"):
        return
    _close_editors(keep=set(_vids()))
    dpg.delete_item("voices_panel", children_only=True)
    for vid in _vids():
        _build_voice_panel(vid)


def build_ui() -> None:
    _register_piano_keys()
    with dpg.window(tag="primary"):
        with dpg.group(horizontal=True):
            # left: controls
            with dpg.child_window(width=400, autosize_y=True):
                _build_session_controls()
                dpg.add_separator()
                dpg.add_text("Voices")
                dpg.add_group(tag="voices_panel")
                _rebuild_voice_panels()
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Add Voice", callback=on_add_voice, width=90)
                    _tip(dpg.last_item(),
                         "Add a new blank voice to the session. Edit its knobs, then "
                         "press Generate to hear it.")
                    dpg.add_combo([], tag="library_combo", width=130)
                    _tip("library_combo", "Voices you've saved to the library (voices/).")
                    dpg.add_button(label="Add Saved", callback=on_add_from_library, width=80)
                    _tip(dpg.last_item(),
                         "Add the voice picked in the list to the current session.")
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
    _refresh_library_combo()
    draw_piano_roll()
    while dpg.is_dearpygui_running():
        frame_update()
        dpg.render_dearpygui_frame()
    save_settings()
    _close_stream()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
