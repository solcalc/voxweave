/* voice.c -- see voice.h. This is the streaming rewrite of synth.py's
 * render_events() inner loop and _pitch_and_amp_mod(). */
#include "voice.h"

/* MIDI note -> Hz (scales.py: midi_to_hz). A=440 at MIDI 69. */
static float midi_to_hz(int midi) {
    return 440.0f * powf(2.0f, (midi - 69) / 12.0f);
}

void voice_init_ee(Voice *v) {
    /* from default_voices()["melody"] */
    v->amp = 0.9f;
    v->tilt_hz = 3200.0f;           /* glottal tilt; de-buzz the saw */
    v->vibrato_hz = 5.7f;
    v->vibrato_semitones = 0.4f;
    v->vibrato_delay = 0.0f;
    v->vibrato_ramp = 0.6f;
    v->scoop_semitones = 0.5f;
    v->scoop_time = 0.08f;
    v->attack_s = 0.06f;

    /* VOWEL_FORMANTS["ee"] = [(270,6,1.0),(2290,10,0.5),(3010,12,0.3)].
     * Q and gain differ per formant; freq sets the resonance. */
    biquad_bandpass(&v->bp[0], 270.0f,  6.0f);  v->fg[0] = 1.00f;
    biquad_bandpass(&v->bp[1], 2290.0f, 10.0f); v->fg[1] = 0.50f;
    biquad_bandpass(&v->bp[2], 3010.0f, 12.0f); v->fg[2] = 0.30f;

    onepole_set(&v->tilt, v->tilt_hz);
    osc_init(&v->osc);
    v->active = 0;
    v->note_pos = 0;
    v->note_len = 0;
    v->base_freq = 0.0f;
}

void voice_note_on(Voice *v, int midi, float dur_s) {
    v->base_freq = midi_to_hz(midi);
    v->note_len = (long)(dur_s * SR);
    v->note_pos = 0;
    v->active = 1;
    env_start(&v->env, v->attack_s, dur_s * 0.9f); /* decay ~ note length */
    /* Note: we deliberately do NOT reset bp[]/tilt filter memory here. */
}

float voice_tick(Voice *v) {
    if (!v->active) return 0.0f;
    if (v->note_pos >= v->note_len) { v->active = 0; return 0.0f; }

    float t = (float)v->note_pos / SR;     /* seconds into the note */

    /* ---- pitch modulation (synth.py: _pitch_and_amp_mod) ----
     * vibrato eases in after `delay` over `ramp` seconds, then a sine LFO;
     * scoop glides up into the pitch (a decaying downward offset). We compute
     * sinf/expf per sample here for clarity; on the real MCU you'd swap these
     * for a recursive sine oscillator + a one-state exp decay (both ~1 mul). */
    float onset = (t - v->vibrato_delay) / (v->vibrato_ramp + 1e-4f);
    if (onset < 0.0f) onset = 0.0f;
    if (onset > 1.0f) onset = 1.0f;
    float lfo     = sinf(TWO_PI * v->vibrato_hz * t);
    float vibrato = v->vibrato_semitones * onset * lfo;
    float scoop   = -v->scoop_semitones * expf(-t / (v->scoop_time + 1e-4f));
    float semis   = vibrato + scoop;
    float freq    = v->base_freq * powf(2.0f, semis / 12.0f);

    /* ---- signal chain (synth.py: render_events body) ----
     * saw -> glottal tilt -> 3 parallel formant bandpasses -> envelope. */
    float sig = osc_tick(&v->osc, freq);
    if (v->tilt_hz > 0.0f) sig = onepole_tick(&v->tilt, sig);

    float shaped = v->fg[0] * biquad_tick(&v->bp[0], sig)
                 + v->fg[1] * biquad_tick(&v->bp[1], sig)
                 + v->fg[2] * biquad_tick(&v->bp[2], sig);

    float out = v->amp * shaped * env_tick(&v->env);
    v->note_pos++;
    return out;
}
