# Desktop build. On the ESP32 you'd build with the ESP-IDF instead (idf.py),
# but the .c/.h files compile unchanged -- only stream.c's output target differs.
CC     = cc
CFLAGS = -O2 -Wall -Wextra -std=c11
LDLIBS = -lm          # math.h (sinf/cosf/expf) lives in libm on desktop

voxweave: stream.c voice.c dsp.h voice.h
	$(CC) $(CFLAGS) stream.c voice.c -o voxweave $(LDLIBS)

run: voxweave
	./voxweave
	@echo "play it with:  aplay out.wav   (or open out.wav in anything)"

clean:
	rm -f voxweave out.wav
