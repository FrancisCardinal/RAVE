import pyaudio
import os
import sys
import wave

PLAYBACK_TO_HEADPHONES = False
WAVE_OUTPUT_FILENAME = "./output/out.wav"
CHANNELS = 8
FREQ = 48000
CHUNK_SIZE = 256

p = pyaudio.PyAudio()

# stream = p.open(
#     format = p.get_sample_size(pyaudio.paFloat32),
#     channels = CHANNELS,
#     rate = FREQ,
#     output = True,
# )

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
wf.setframerate(FREQ)

print("* recording")
data = 1
while data:
    data = sys.stdin.buffer.read(CHUNK_SIZE)
    wf.writeframes(data)
    if PLAYBACK_TO_HEADPHONES:
        stream.write(data)

print("* done recording")
wf.close()
# stream.terminate()