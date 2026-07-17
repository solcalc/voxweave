/* esp32_arduino.ino -- the REAL-hardware twin of stream.c, for Arduino.
 *
 * This is what actually runs on the ESP32-WROOM + MAX98357A. Compared to the
 * desktop stream.c, only the plumbing changes:
 *
 *   stream.c (desktop)          esp32_arduino.ino (hardware)
 *   ------------------          -----------------------------
 *   ping[]/pong[] arrays   -->  ONE work buffer `block[]` (the driver keeps its
 *                               own pool of DMA buffers -- you don't declare it)
 *   fwrite(buf, ...)       -->  i2s_write(...)  (copies block[] into a free DMA
 *                               buffer; BLOCKS if the pool is full = your pacing)
 *   main()                -->  setup() + loop()
 *
 * The synthesis files are IDENTICAL to the desktop demo: drop voice.c, voice.h,
 * and dsp.h into this sketch folder next to the .ino and they compile as-is.
 * (Arduino builds every .c/.cpp/.h in the sketch folder along with the .ino.)
 *
 * NOTE: uses the legacy `driver/i2s.h` API -- the one every MAX98357A tutorial
 * uses, and where dma_buf_count/dma_buf_len map directly to "the DMA buffers"
 * we've been talking about. On ESP32 Arduino core 3.x it still compiles (may
 * warn as deprecated); the modern replacement is the ESP_I2S `I2SClass`, but
 * the concepts below are unchanged. Untested on my end -- wiring/pins are the
 * common MAX98357A layout; adjust GPIOs to your board.
 */
#include <driver/i2s.h>
#include "voice.h"     // <-- same file as the desktop demo

// ---- MAX98357A wiring (I2S DAC + amp in one). Pick any free GPIOs. ----
#define PIN_BCLK  26   // bit clock  -> MAX98357A BCLK
#define PIN_LRC   25   // word/LR clk -> MAX98357A LRC
#define PIN_DOUT  22   // data        -> MAX98357A DIN
// (MAX98357A also needs VIN, GND, and its GAIN/SD pins set per its datasheet.)

#define BLOCK 256      // samples we synthesize per i2s_write, same as stream.c

static Voice v;
static int16_t block[BLOCK];   // OUR work buffer (compare: ping/pong in stream.c)

// Toy note generator, same pentatonic phrase as the desktop demo.
static const int PHRASE[] = { 64, 67, 69, 72, 69, 67, 64, 60 };
static int phrase_i = 0;

static inline int16_t to_pcm(float x) {
  x = tanhf(x);                       // master soft-limiter (synth.py: np.tanh)
  int val = (int)(x * 32767.0f);
  if (val >  32767) val =  32767;
  if (val < -32768) val = -32768;
  return (int16_t)val;
}

void setup() {
  voice_init_ee(&v);

  // ---- Describe the DMA buffer pool to the driver. THIS is where the buffers
  // we keep talking about get created -- in the ESP32's internal SRAM, owned
  // by the driver. We never see their pointers; we just say how many/how big.
  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = (int)SR,                          // from dsp.h (22050)
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,     // mono -> MAX98357A
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,        // <-- HOW MANY DMA buffers (the pool). 8 is safe.
    .dma_buf_len   = BLOCK,    // <-- samples per DMA buffer.
    .use_apll = false,
    .tx_desc_auto_clear = true,
    .fixed_mclk = 0,
  };
  i2s_pin_config_t pins = {
    .bck_io_num   = PIN_BCLK,
    .ws_io_num    = PIN_LRC,
    .data_out_num = PIN_DOUT,
    .data_in_num  = I2S_PIN_NO_CHANGE,   // we only transmit
  };

  i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);  // allocates the DMA buffers
  i2s_set_pin(I2S_NUM_0, &pins);
}

void loop() {
  // 1) FILL our work buffer -- identical to fill_block() in stream.c.
  for (int i = 0; i < BLOCK; i++) {
    if (!v.active) {
      int midi = PHRASE[phrase_i];
      phrase_i = (phrase_i + 1) % (int)(sizeof(PHRASE) / sizeof(PHRASE[0]));
      voice_note_on(&v, midi, 0.7f);
    }
    block[i] = to_pcm(voice_tick(&v));
  }

  // 2) SHIP it. i2s_write copies block[] into the next FREE DMA buffer, then
  // the DMA engine streams that buffer to the DAC on its own. If every DMA
  // buffer is still full (we got ahead), this call BLOCKS until one frees up --
  // that block is what paces loop() to the speaker's clock. (This is the line
  // that was `fwrite` in stream.c.)
  size_t written = 0;
  i2s_write(I2S_NUM_0, block, sizeof(block), &written, portMAX_DELAY);
}
