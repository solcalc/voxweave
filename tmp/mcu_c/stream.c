/* stream.c -- THE MAIN LESSON. How audio actually leaves the chip.
 *
 * This file simulates, on your desktop, exactly what the ESP32-WROOM does in
 * hardware. The synthesis code (dsp.h / voice.c) is byte-for-byte what you'd
 * flash. Only this file is "fake": instead of an I2S peripheral clocking samples
 * to a MAX98357A, we clock them into a .wav file. Same shape, visible output.
 *
 * ================= THE MENTAL MODEL =================
 *
 * On the real board there are TWO clocks running at once:
 *
 *   THE SPEAKER'S CLOCK (hardware, never stops):
 *     The I2S peripheral pulls samples out of a small DMA buffer at exactly
 *     SR samples/sec and shifts them to the DAC. The CPU does NOTHING per
 *     sample -- this is dedicated silicon. It cannot wait for you. If the
 *     buffer it's reading runs dry, you hear a pop (an "underrun").
 *
 *   YOUR CLOCK (software, must stay ahead):
 *     You keep TWO buffers (ping & pong). While I2S drains one, you fill the
 *     other with voice_tick(). When I2S finishes a buffer it flips to the
 *     other and hands the drained one back to you to refill. Round and round.
 *
 * You are never "storing the song." At any instant only BLOCK*2 samples exist
 * in RAM -- about 23 ms at 22.05kHz with BLOCK=256. The song is infinite; you
 * only ever hold the next couple blocks.
 *
 * The whole game: fill the next block before I2S drains the current one. From
 * our earlier budget that's ~15-35% of one core, so there's lots of slack.
 * ===================================================
 */
#include <stdio.h>
#include <stdint.h>
#include "voice.h"

#define BLOCK      256    /* samples per buffer. Smaller = less latency, tighter
                          * deadline. 256 @ 22.05k = 11.6 ms per block. */
#define NUM_BLOCKS 800    /* how many blocks to render in this demo (~9.3 s). On
                          * the MCU this loop is `while(1)` -- endless mode. */

/* ---- The two DMA buffers. On the ESP32 these live in DMA-capable SRAM and
 * you'd register them with the I2S driver. Here they're just arrays. ---- */
static int16_t ping[BLOCK];
static int16_t pong[BLOCK];

/* ---- A tiny stand-in for the note generator (generator.py / arrangement.py).
 * The real one produces NoteEvents from the seeded PRNG + scale. Here we just
 * walk a little pentatonic phrase so you can hear the streaming work. ---- */
static const int PHRASE[] = { 64, 67, 69, 72, 69, 67, 64, 60 };
static int phrase_i = 0;

/* Master soft-limiter (synth.py: np.tanh on the bus). We CAN'T peak-normalize a
 * stream we haven't finished (that's what mix() does offline), so instead we
 * tanh each sample: transparent when quiet, folds peaks under 1.0 when loud. */
static inline int16_t to_pcm(float x) {
    x = tanhf(x);                         /* soft clip to [-1,1] */
    int v = (int)(x * 32767.0f);
    if (v >  32767) v =  32767;
    if (v < -32768) v = -32768;
    return (int16_t)v;
}

/* Fill ONE block by ticking the voice BLOCK times. THIS is the function that,
 * on the MCU, must finish before I2S drains the other buffer. It is the exact
 * streaming equivalent of synth.py's per-sample render loop. */
static void fill_block(int16_t *buf, Voice *v) {
    for (int i = 0; i < BLOCK; i++) {
        /* If the current note ended, ask the "generator" for the next one.
         * (On the MCU this decision comes off a queue filled by core 0.) */
        if (!v->active) {
            int midi = PHRASE[phrase_i];
            phrase_i = (phrase_i + 1) % (int)(sizeof(PHRASE)/sizeof(PHRASE[0]));
            voice_note_on(v, midi, 0.7f);   /* 0.7s notes, legato-ish */
        }
        buf[i] = to_pcm(voice_tick(v));
    }
}

/* ---- Minimal WAV writer so the demo is audible. NOT part of the MCU port;
 * on hardware, fill_block's buffer goes to i2s_write() instead of fwrite(). */
static void write_wav_header(FILE *f, int nsamples) {
    int32_t rate = (int32_t)SR, byte_rate = rate * 2;
    int32_t data_bytes = nsamples * 2, riff = 36 + data_bytes;
    fwrite("RIFF", 1, 4, f); fwrite(&riff, 4, 1, f); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    int32_t fmt_len = 16; int16_t pcm = 1, ch = 1, bits = 16, align = 2;
    fwrite(&fmt_len,4,1,f); fwrite(&pcm,2,1,f); fwrite(&ch,2,1,f);
    fwrite(&rate,4,1,f); fwrite(&byte_rate,4,1,f);
    fwrite(&align,2,1,f); fwrite(&bits,2,1,f);
    fwrite("data",1,4,f); fwrite(&data_bytes,4,1,f);
}

int main(void) {
    Voice v;
    voice_init_ee(&v);

    FILE *f = fopen("out.wav", "wb");
    if (!f) { perror("fopen"); return 1; }
    write_wav_header(f, NUM_BLOCKS * BLOCK);

    /* THE PING-PONG LOOP. Read it as: "fill a buffer, ship it, repeat."
     * On hardware, `fwrite` is replaced by i2s_write(), which blocks until the
     * DMA engine has a free buffer -- that block is precisely what keeps YOUR
     * clock synced to the SPEAKER'S clock. Here fwrite is instant, so the WAV
     * is written as fast as possible; the audio inside is identical. */
    int use_ping = 1;
    for (int b = 0; b < NUM_BLOCKS; b++) {
        int16_t *buf = use_ping ? ping : pong;   /* pick the free buffer */
        fill_block(buf, &v);                     /* <-- the real-time work */
        fwrite(buf, sizeof(int16_t), BLOCK, f);  /* <-- i2s_write() on MCU */
        use_ping = !use_ping;                    /* flip ping<->pong */
    }

    fclose(f);
    printf("wrote out.wav  (%d blocks x %d samples @ %.0f Hz = %.1f s)\n",
           NUM_BLOCKS, BLOCK, SR, NUM_BLOCKS * BLOCK / SR);
    printf("only %d samples (2 x BLOCK) ever lived in RAM at once.\n", 2*BLOCK);
    return 0;
}
