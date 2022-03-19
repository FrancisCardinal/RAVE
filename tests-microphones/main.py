import pyaudio
import os
import sys
import wave
WAVE_OUTPUT_FILENAME = "./output/out.wav"

channels = 8
freq = 48000
chunkSize = 256
RECORD_SECONDS = 4

p = pyaudio.PyAudio()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
wf.setframerate(freq)

print("* recording")
data = 1
while data:
    data = sys.stdin.buffer.read(chunkSize)
    wf.writeframes(data)

print("* done recording")
wf.close()
