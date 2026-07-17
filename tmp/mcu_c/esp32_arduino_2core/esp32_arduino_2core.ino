/* esp32_arduino_2core.ino -- the robust layout, still 100% Arduino API.
 *
 * Proves the point: task pinning is a FreeRTOS feature, and Arduino-ESP32 IS
 * FreeRTOS. No idf.py needed. We split the work across the WROOM's two cores:
 *
 *   CORE 1 (audioTask)      : audio ONLY. Fills a block, calls i2s_write, forever.
 *                             Never blocked by WiFi/generator, so no underruns.
 *   CORE 0 (generatorTask)  : decides which notes to sing (your generator.py /
 *                             arrangement.py logic) and hands them to core 1.
 *                             This is also the core WiFi/BT run on.
 *
 * They talk through a FreeRTOS QUEUE -- a lock-free, ISR-safe ring the audio
 * task pops from and the generator pushes to. The audio task NEVER waits on the
 * generator (it plays silence if the queue is momentarily empty), which is what
 * keeps the speaker glitch-free.
 *
 * Same synthesis files (dsp.h / voice.c) as every other version.
 */
#include <driver/i2s.h>
#include "voice.h"

#define PIN_BCLK 26
#define PIN_LRC  25
#define PIN_DOUT 22
#define BLOCK    256

// A note the generator hands to the synth: "sing this MIDI pitch for this long."
typedef struct { int midi; float dur; } NoteMsg;

static QueueHandle_t note_q;    // core0 --> core1 channel (depth 8 notes ahead)

static inline int16_t to_pcm(float x) {
  x = tanhf(x);
  int val = (int)(x * 32767.0f);
  if (val >  32767) val =  32767;
  if (val < -32768) val = -32768;
  return (int16_t)val;
}

/* ------------------------- CORE 1: the audio task ------------------------- */
static void audioTask(void *arg) {
  Voice v;
  voice_init_ee(&v);
  int16_t block[BLOCK];

  for (;;) {                                   // == the endless ping-pong loop
    for (int i = 0; i < BLOCK; i++) {
      if (!v.active) {
        NoteMsg m;
        // Try to grab the next note WITHOUT blocking (timeout 0). If none is
        // ready we simply emit silence this sample -- audio must never stall.
        if (xQueueReceive(note_q, &m, 0) == pdTRUE)
          voice_note_on(&v, m.midi, m.dur);
      }
      block[i] = to_pcm(voice_tick(&v));
    }
    size_t written;
    // Blocks until a DMA buffer frees up -> paces this task to the DAC clock.
    i2s_write(I2S_NUM_0, block, sizeof(block), &written, portMAX_DELAY);
  }
}

/* ----------------------- CORE 0: the generator task ----------------------- */
static void generatorTask(void *arg) {
  static const int PHRASE[] = { 64, 67, 69, 72, 69, 67, 64, 60 };
  int i = 0;
  for (;;) {
    NoteMsg m = { PHRASE[i], 0.7f };
    i = (i + 1) % (int)(sizeof(PHRASE) / sizeof(PHRASE[0]));
    // Block here if the queue is full -- fine: the generator is allowed to wait
    // for the synth to catch up. (This is where generator.py's logic goes.)
    xQueueSend(note_q, &m, portMAX_DELAY);
  }
}

void setup() {
  // I2S / DMA-buffer setup, identical to the single-core sketch.
  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = (int)SR,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len   = BLOCK,
    .use_apll = false,
    .tx_desc_auto_clear = true,
    .fixed_mclk = 0,
  };
  i2s_pin_config_t pins = {
    .bck_io_num = PIN_BCLK, .ws_io_num = PIN_LRC,
    .data_out_num = PIN_DOUT, .data_in_num = I2S_PIN_NO_CHANGE,
  };
  i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pins);

  note_q = xQueueCreate(8, sizeof(NoteMsg));   // 8 notes of look-ahead

  // The star of the show: last arg is the core number.
  //   xTaskCreatePinnedToCore(fn, name, stack, arg, priority, handle, CORE);
  xTaskCreatePinnedToCore(audioTask,     "audio", 4096, NULL, 3, NULL, 1); // core 1
  xTaskCreatePinnedToCore(generatorTask, "gen",   4096, NULL, 1, NULL, 0); // core 0
}

void loop() {
  // Nothing here. Arduino's loopTask (itself on core 1) just idles; all the
  // real work lives in the two tasks above. You could delete loop() logic
  // entirely, or use it for slow housekeeping.
  vTaskDelay(pdMS_TO_TICKS(1000));
}
