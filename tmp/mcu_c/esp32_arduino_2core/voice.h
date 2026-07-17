/* voice.h -- one "doll": the streaming twin of synth.py's render_events().
 *
 * synth.py builds a whole note buffer per event, then sums buffers. We can't
 * hold whole buffers, so a Voice is a persistent bundle of the primitive state
 * machines from dsp.h plus a little modulation state. You:
 *   1) voice_init() once at boot,
 *   2) voice_note_on() when the generator says "sing MIDI note X for D seconds",
 *   3) call voice_tick() once per output sample forever; it returns silence
 *      between notes and the shaped tone during them.
 *
 * This demo implements ONE sung voice (the "ee" formant tone with vibrato,
 * scoop, tilt). Harmony/drone/opera are the same struct with different config;
 * percussion is a parallel variant. Kept to one voice so the mechanism is
 * legible -- the point is the streaming shape, not feature-completeness.
 */
#ifndef VOICE_H
#define VOICE_H

#include "dsp.h"

typedef struct {
    /* --- fixed config (VoiceConfig in config.py), baked at init --- */
    float amp;
    float tilt_hz;              /* one-pole cutoff; 0 = bypass */
    float vibrato_hz, vibrato_semitones, vibrato_delay, vibrato_ramp;
    float scoop_semitones, scoop_time;
    float attack_s;             /* default decay comes per-note */
    float fg[3];                /* formant output gains F1,F2,F3 */

    /* --- signal-chain state (the dsp.h machines) --- */
    Osc     osc;
    OnePole tilt;
    Biquad  bp[3];              /* three parallel bandpass formants */
    Env     env;

    /* --- per-note modulation state --- */
    int   active;               /* 1 while a note is sounding */
    long  note_pos;             /* samples since note_on (drives vibrato/scoop) */
    long  note_len;             /* total samples this note lasts */
    float base_freq;            /* the note's target pitch in Hz */
} Voice;

/* Configure a voice to sing the female "ee" vowel (numbers lifted straight from
 * config.py VOWEL_FORMANTS["ee"] and default_voices()["melody"]). */
void voice_init_ee(Voice *v);

/* Begin a note. `midi` is a MIDI note number, `dur_s` its length in seconds.
 * Resets the envelope + modulation but NOT the filter memory (a real singer's
 * resonators don't teleport to zero between notes -- and it avoids clicks). */
void voice_note_on(Voice *v, int midi, float dur_s);

/* Produce exactly one output sample. Call at SR forever. */
float voice_tick(Voice *v);

#endif /* VOICE_H */
