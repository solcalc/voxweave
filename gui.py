#!/usr/bin/env python3
"""Desktop GUI for the singing-doll generative music testbed (Dear PyGui).

A thin shell over the existing pipeline -- it edits every session/voice parameter,
toggles voices on/off, renders via generate_voice -> render_events -> mix, and
draws each voice as a read-only piano roll. Nothing in the generator/synth core
is modified; this module only reads from it.

    python gui.py
"""

import ast
import dataclasses
import json
import os
import random
import threading
import time

import dearpygui.dearpygui as dpg

import generator
import synth
import wav_io
from arrangement import build_arrangement, flat_arrangement
from config import SessionConstitution, VoiceConfig, VOWEL_FORMANTS, default_voices
from scales import NOTE_CLASS, SCALES

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

# Where widget values are remembered between sessions.
SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui_settings.json")

# Session fields shown in the top panel (name -> widget kind). key/scale are combos.
SESSION_INT_FIELDS = (
    "tempo", "seed", "bars", "beats_per_bar",
    "sample_rate", "phrase_bars", "chord_bars",
)

# Hover help, written for a non-musician: what it does, in plain words, plus what
# kind of value to enter.
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

# Live state, populated on each Generate.
state = {
    "voices": default_voices(),          # {name: VoiceConfig}, source of truth
    "const": None,                        # last SessionConstitution
    "events": {},                         # {name: [NoteEvent]}
    "mixed": None,                        # last mixed numpy buffer
    "geom": None,                         # piano-roll plot geometry for the playhead
    # --- background generation ---
    "gen_running": False,
    "gen_cancel": False,
    "gen_progress": 0.0,                  # 0..1
    "gen_status": "",
    "gen_result": None,                   # (const, events, mixed) handed back to main thread
    # --- playback / playhead ---
    "playing": False,
    "play_start": 0.0,                    # perf_counter at play onset
    "play_dur": 0.0,                      # seconds
}


# --------------------------------------------------------------------------- #
# Flatten (parity with testbed.py --flat): zero the structural knobs.
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
        steps = len(enabled) + 1
        state["gen_progress"] = 1.0 / steps
        events, buffers = {}, []
        for i, (name, cfg) in enumerate(enabled):
            if state["gen_cancel"]:
                state["gen_status"] = "cancelled"
                state["gen_running"] = False
                return
            state["gen_status"] = f"rendering {name}..."
            evs = generator.generate_voice(const, cfg, arr)
            events[name] = evs
            buffers.append(synth.render_events(evs, const, cfg))
            state["gen_progress"] = (i + 2) / steps
        mixed = synth.mix(buffers) if buffers else None
        state["gen_result"] = (const, events, mixed)
    except Exception as exc:  # surface, don't crash the render loop
        state["gen_status"] = f"error: {exc}"
    finally:
        state["gen_running"] = False


def on_generate() -> None:
    if state["gen_running"]:
        return
    # roll a fresh seed if requested, writing it back so the field shows what was used
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
        state["mixed"] = None
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


def on_play() -> None:
    if state["mixed"] is None:
        _status("generate first")
        return
    try:
        import numpy as np
        import sounddevice as sd
    except Exception as exc:  # ImportError or PortAudio missing
        _status(f"playback unavailable ({exc})")
        return
    buf = np.clip(state["mixed"], -1.0, 1.0).astype(np.float32)
    sd.play(buf, state["const"].sample_rate)
    state["playing"] = True
    state["play_start"] = time.perf_counter()
    state["play_dur"] = len(buf) / state["const"].sample_rate
    _status("playing...")


def on_stop() -> None:
    state["playing"] = False
    try:
        import sounddevice as sd
    except Exception:
        return
    sd.stop()
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
    tags = ["session_key", "session_scale", "session_seed_random", "session_flat"]
    tags += [f"session_{n}" for n in SESSION_INT_FIELDS]
    for name in VOICE_NAMES:
        tags.append(_enabled_tag(name))
        for f in dataclasses.fields(VoiceConfig):
            if f.name != "name":
                tags.append(_voice_tag(name, f.name))
    return tags


def save_settings() -> None:
    data = {tag: dpg.get_value(tag) for tag in _setting_tags() if dpg.does_item_exist(tag)}
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
                pass  # stale/incompatible entry; ignore


def _default_settings() -> dict:
    """The value every widget should hold in its factory-default state."""
    const = SessionConstitution()
    voices = default_voices()
    out = {
        "session_key": const.key,
        "session_scale": const.scale,
        "session_seed_random": True,
        "session_flat": False,
    }
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
    # reset the config source-of-truth too, so any prior flatten() is cleared
    state["voices"] = default_voices()
    for tag, val in _default_settings().items():
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, val)
    _status("reset to defaults")


# --------------------------------------------------------------------------- #
# Piano roll.
# --------------------------------------------------------------------------- #
def draw_piano_roll() -> None:
    dpg.delete_item("piano_roll", children_only=True)
    const = state["const"]
    events = state["events"]

    plot_x0, plot_y0 = MARGIN, MARGIN
    plot_w = CANVAS_W - 2 * MARGIN
    plot_h = CANVAS_H - 2 * MARGIN

    # background + frame
    dpg.draw_rectangle((plot_x0, plot_y0), (plot_x0 + plot_w, plot_y0 + plot_h),
                       fill=(28, 28, 32), color=(70, 70, 80), parent="piano_roll")

    all_events = [ev for evs in events.values() for ev in evs]
    if not all_events or const is None:
        state["geom"] = None
        dpg.draw_text((plot_x0 + 12, plot_y0 + 12), "no notes -- press Generate",
                      size=18, color=(160, 160, 160), parent="piano_roll")
        return

    total_secs = max(const.total_seconds, 1e-6)
    # geometry the playhead uses to map elapsed time -> x
    state["geom"] = {"x0": plot_x0, "w": plot_w, "y0": plot_y0, "h": plot_h,
                     "total_secs": total_secs}
    minp = min(ev.midi for ev in all_events)
    maxp = max(ev.midi for ev in all_events)
    minp, maxp = minp - 2, maxp + 2
    span = max(maxp - minp, 1)
    row_h = plot_h / (span + 1)

    def x_of(t):
        return plot_x0 + (t / total_secs) * plot_w

    def y_of(midi):
        return plot_y0 + ((maxp - midi) / span) * (plot_h - row_h)

    # bar lines + numbers
    for bar in range(const.bars + 1):
        x = x_of(bar * const.sec_per_bar)
        dpg.draw_line((x, plot_y0), (x, plot_y0 + plot_h), color=(55, 55, 62),
                      parent="piano_roll")
        if bar < const.bars:
            dpg.draw_text((x + 3, plot_y0 + plot_h - 18), str(bar), size=13,
                          color=(90, 90, 100), parent="piano_roll")

    # notes: draw the audible-onset length (duration minus the long decay tail)
    for name in VOICE_NAMES:
        if name not in events:
            continue
        color = VOICE_COLORS[name]
        cfg = state["voices"][name]
        for ev in events[name]:
            vis = max(0.08, ev.duration - cfg.decay)
            x0, x1 = x_of(ev.start), x_of(min(ev.start + vis, total_secs))
            y0 = y_of(ev.midi)
            dpg.draw_rectangle((x0, y0), (max(x1, x0 + 2), y0 + row_h * 0.9),
                               fill=color, color=color, parent="piano_roll")

    # legend
    for i, name in enumerate(VOICE_NAMES):
        on = dpg.get_value(_enabled_tag(name))
        c = VOICE_COLORS[name] if (name in events) else (90, 90, 90)
        lx = plot_x0 + 12 + i * 120
        dpg.draw_rectangle((lx, plot_y0 + 8), (lx + 14, plot_y0 + 22),
                           fill=c, color=c, parent="piano_roll")
        dpg.draw_text((lx + 20, plot_y0 + 7), name if on else f"{name} (off)",
                      size=14, color=(200, 200, 200), parent="piano_roll")


def _draw_playhead() -> None:
    """Moving vertical line at the current playback position (main thread only)."""
    if dpg.does_item_exist("playhead"):
        dpg.delete_item("playhead")
    geom = state["geom"]
    if not state["playing"] or geom is None:
        return
    elapsed = time.perf_counter() - state["play_start"]
    if elapsed >= state["play_dur"]:
        state["playing"] = False
        return
    frac = elapsed / geom["total_secs"] if geom["total_secs"] else 0.0
    x = geom["x0"] + min(frac, 1.0) * geom["w"]
    dpg.draw_line((x, geom["y0"]), (x, geom["y0"] + geom["h"]),
                  color=(255, 70, 70), thickness=2, tag="playhead", parent="piano_roll")


def frame_update() -> None:
    """Called once per rendered frame to reflect background work + playback."""
    # generation progress -> bar + status
    if state["gen_running"]:
        dpg.set_value("gen_progress_bar", state["gen_progress"])
        if state["gen_status"]:
            _status(state["gen_status"])
    # hand a finished render back to the UI thread
    if state["gen_result"] is not None:
        const, events, mixed = state["gen_result"]
        state["gen_result"] = None
        state["const"], state["events"], state["mixed"] = const, events, mixed
        dpg.set_value("gen_progress_bar", 1.0)
        draw_piano_roll()
        secs = len(mixed) / const.sample_rate if mixed is not None else 0.0
        _status(f"generated {len(events)} voice(s), {secs:.1f}s")
    elif not state["gen_running"] and state["gen_status"] == "cancelled":
        state["gen_status"] = ""
        dpg.set_value("gen_progress_bar", 0.0)
        _status("cancelled")

    _draw_playhead()


# --------------------------------------------------------------------------- #
# UI construction.
# --------------------------------------------------------------------------- #
def _tip(tag: str, text: str) -> None:
    """Attach a hover tooltip to a widget."""
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
    # enable checkbox sits at the header level, so a voice can be toggled without
    # expanding its parameter list
    with dpg.group(horizontal=True):
        dpg.add_checkbox(tag=_enabled_tag(name), default_value=True)
        _tip(_enabled_tag(name), VOICE_ENABLED_HELP)
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
                    # double (not input_float) so untouched values keep full precision
                    # and match the CLI byte-for-byte; the generator is float-sensitive.
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
                with dpg.group(horizontal=True):
                    dpg.add_progress_bar(tag="gen_progress_bar", width=300, default_value=0.0)
                    dpg.add_button(label="Cancel", callback=on_cancel_gen, width=90)
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
    load_settings()  # restore last session's values
    draw_piano_roll()
    # manual render loop so the playhead + progress bar animate each frame
    while dpg.is_dearpygui_running():
        frame_update()
        dpg.render_dearpygui_frame()
    save_settings()  # remember values for next time
    dpg.destroy_context()


if __name__ == "__main__":
    main()
