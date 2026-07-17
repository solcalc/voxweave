/* dsp.h -- the synthesis primitives, ported from synth.py to float32 C.
 *
 * READ THIS FIRST if you're trying to understand the port. Every function here
 * has a direct twin in synth.py. The ONE big difference from the Python:
 *
 *   Python renders a whole note (n samples) at once into a numpy array.
 *   Here, each primitive is a little STATE MACHINE that produces ONE sample at
 *   a time when you call its _tick(). That's the change that makes realtime
 *   streaming possible -- we can stop after any sample and resume later.
 *
 * Why one-sample-at-a-time? Because on the MCU we don't have room to hold the
 * whole song. We fill tiny buffers just ahead of the speaker (see stream.c).
 * A recursive filter (biquad, one-pole) is ALREADY sample-at-a-time in Python
 * (that's the `for n in range(...)` loop); we're just exposing that loop body.
 *
 * float32, not float64: the ESP32-WROOM's FPU is single-precision hardware.
 * `double` would be emulated in software (~10-20x slower), so we use `float`
 * everywhere. Audio doesn't need 15 digits of mantissa.
 */
#ifndef DSP_H
#define DSP_H

#include <math.h>   /* sinf/cosf/expf -- on the ESP32 these are cheap and only
                     * called at note-start (coefficient setup), not per sample. */

#define SR       22050.0f   /* sample rate. 22.05k is plenty for pads/formants
                             * and halves the CPU vs 44.1k. Bump to 44100 later
                             * if you want -- nothing else changes. */
#define TWO_PI   6.28318530717958647692f

/* ------------------------------------------------------------------ *
 * Phase-accumulator oscillator  (synth.py: sawtooth / sawtooth_fm)
 * ------------------------------------------------------------------ *
 * `phase` walks 0..1 and wraps. Each sample we add `inc = freq/SR`.
 * To do vibrato/scoop we just change `inc` every sample -- exactly the
 * "increment register updated from the LFO" comment in synth.py.
 */
typedef struct {
    float phase;   /* current position in the waveform, 0..1 */
} Osc;

static inline void osc_init(Osc *o) { o->phase = 0.0f; }

/* Return one sawtooth sample in [-1,1] for the given instantaneous freq. */
static inline float osc_tick(Osc *o, float freq_hz) {
    o->phase += freq_hz * (1.0f / SR);
    if (o->phase >= 1.0f) o->phase -= 1.0f;   /* wrap (cheaper than fmod) */
    return 2.0f * o->phase - 1.0f;            /* ramp 0..1 -> -1..1 */
}

/* ------------------------------------------------------------------ *
 * One-pole lowpass  (synth.py: one_pole_lp / _one_pole_lp_kernel)
 * ------------------------------------------------------------------ *
 * y[n] = b*x[n] + a*y[n-1].  The glottal spectral tilt that de-buzzes
 * the saw. State is a single previous-output value.
 */
typedef struct {
    float a, b;   /* coefficients, baked once from cutoff */
    float y1;     /* y[n-1] */
} OnePole;

static inline void onepole_set(OnePole *p, float cutoff_hz) {
    p->a = expf(-TWO_PI * cutoff_hz / SR);
    p->b = 1.0f - p->a;
    p->y1 = 0.0f;
}

static inline float onepole_tick(OnePole *p, float x) {
    p->y1 = p->b * x + p->a * p->y1;
    return p->y1;
}

/* ------------------------------------------------------------------ *
 * Bandpass biquad  (synth.py: class Biquad / _biquad_kernel)
 * ------------------------------------------------------------------ *
 * RBJ cookbook, constant-0dB-peak-gain bandpass, Direct Form I.
 * This is the line the Python comment promised ports "line-for-line":
 *   y = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2;
 * State is the last two inputs (x1,x2) and last two outputs (y1,y2).
 */
typedef struct {
    float b0, b1, b2, a1, a2;   /* coefficients (a0 normalized to 1) */
    float x1, x2, y1, y2;       /* filter memory */
} Biquad;

static inline void biquad_bandpass(Biquad *f, float f0, float q) {
    float w0    = TWO_PI * f0 / SR;
    float cw0   = cosf(w0);
    float sw0   = sinf(w0);
    float alpha = sw0 / (2.0f * q);
    float a0    = 1.0f + alpha;
    f->b0 =  alpha / a0;
    f->b1 =  0.0f;
    f->b2 = -alpha / a0;
    f->a1 = (-2.0f * cw0) / a0;
    f->a2 = ( 1.0f - alpha) / a0;
    f->x1 = f->x2 = f->y1 = f->y2 = 0.0f;
}

static inline float biquad_tick(Biquad *f, float x) {
    float y = f->b0*x + f->b1*f->x1 + f->b2*f->x2 - f->a1*f->y1 - f->a2*f->y2;
    f->x2 = f->x1; f->x1 = x;   /* shift input history */
    f->y2 = f->y1; f->y1 = y;   /* shift output history */
    return y;
}

/* ------------------------------------------------------------------ *
 * Exponential-decay envelope  (synth.py: adsr_envelope)
 * ------------------------------------------------------------------ *
 * Linear attack ramp to 1.0, then exponential decay. In Python this was
 * built with np.linspace / np.exp over the whole note; as a streaming
 * state machine it's two counters and one multiply-per-sample decay.
 */
typedef struct {
    int   attack_samps;   /* how many samples the attack ramp lasts */
    int   pos;            /* samples elapsed since note start */
    float decay_mult;     /* per-sample multiplier during decay (exp) */
    float level;          /* current envelope value */
} Env;

static inline void env_start(Env *e, float attack_s, float decay_s) {
    e->attack_samps = (int)(attack_s * SR);
    e->pos = 0;
    /* exp decay with time-constant `decay_s`: level *= e^(-1/(decay*SR)) each
     * sample reproduces synth.py's np.exp(-t/decay). */
    e->decay_mult = expf(-1.0f / (decay_s * SR + 1e-4f));
    e->level = 0.0f;
}

static inline float env_tick(Env *e) {
    if (e->pos < e->attack_samps) {
        e->level = (float)e->pos / (float)e->attack_samps;   /* linear ramp up */
    } else {
        e->level *= e->decay_mult;                           /* exp decay */
    }
    e->pos++;
    return e->level;
}

#endif /* DSP_H */
