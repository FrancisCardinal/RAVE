# import pyaudio
# import os
# import sys
# import wave

# PLAYBACK_TO_HEADPHONES = False
# WAVE_OUTPUT_FILENAME = "./output/out.wav"
# CHANNELS = 8
# FREQ = 48000
# CHUNK_SIZE = 256

# p = pyaudio.PyAudio()

# # stream = p.open(
# #     format = p.get_sample_size(pyaudio.paFloat32),
# #     channels = CHANNELS,
# #     rate = FREQ,
# #     output = True,
# # )

# wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
# wf.setnchannels(CHANNELS)
# wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
# wf.setframerate(FREQ)

# print("* recording")
# data = 1
# while data:
#     data = sys.stdin.buffer.read(CHUNK_SIZE)
#     wf.writeframes(data)
#     if PLAYBACK_TO_HEADPHONES:
#         stream.write(data)

# print("* done recording")
# wf.close()
# # stream.terminate()

# import pyaudio

# p = pyaudio.PyAudio()
# info = p.get_host_api_info_by_index(0)
# numdevices = info.get('deviceCount')
# #for each audio device, determine if is an input or an output and add it to the appropriate list and dictionary
# for i in range (0,numdevices):
#         if p.get_device_info_by_host_api_device_index(0,i).get('maxInputChannels')>0:
#                 print(f"Input Device id {i},- {p.get_device_info_by_host_api_device_index(0,i).get('name')}")

#         if p.get_device_info_by_host_api_device_index(0,i).get('maxOutputChannels')>0:
#                 print(f"Outut Device id {i},- {p.get_device_info_by_host_api_device_index(0,i).get('name')}")

# devinfo = p.get_device_info_by_index(1)
# print(f"Selected device is ,{devinfo.get('name')}")
# # if p.is_format_supported(44100.0,  # Sample rate
# #                          input_device=devinfo["index"],
# #                          input_channels=devinfo['maxInputChannels'],
# #                          input_format=pyaudio.paInt16):
# #   print('Yay!')
# p.terminate()

import pyodas.io as io
import pyodas.utils as utils
import wave
import time
import pyaudio
import numpy as np
import os

print(os.getcwd())

CHANNELS = 8
FREQ = 48000
CHUNK = 256

# # instantiate PyAudio (1)
# p = pyaudio.PyAudio()

# info = p.get_host_api_info_by_index(0)
# numdevices = info.get('deviceCount')
# # for each audio device, determine if is an input or an output and add it to the appropriate list and dictionary
# for i in range (0,numdevices):
#         if p.get_device_info_by_host_api_device_index(0,i).get('maxInputChannels')>0:
#                 print(f"Input Device id {i},- {p.get_device_info_by_host_api_device_index(0,i).get('name')}")

#         if p.get_device_info_by_host_api_device_index(0,i).get('maxOutputChannels')>0:
#                 print(f"Outut Device id {i},- {p.get_device_info_by_host_api_device_index(0,i).get('name')}")


file_params = (CHANNELS,
               2,
               48000,
               0,
               'NONE',
               'NONE')

# mic = io.MicSource(CHANNELS, mic_index=2)
# headphone = io.PlaybackSink(2, device_index=2)
# mic = io.MicSource(CHANNELS, mic_index=2)
# wav_source = io.WavSource("./clap_2_channels.wav")
wav_sink = io.WavSink("./test.wav", file_params)

# start = time.time()
# play stream (3)
start = time.time()
while True:
    data = np.ones((2,256), dtype=utils.TYPES.TIME)
    data *= 34000

    if data is None:
        break
    wav_sink(data)
    # headphone(data)
