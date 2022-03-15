from pyodas.io import WavSource, MicSource, PlaybackSink, WavSink
from pyodas.utils import CONST, load_mic_array_from_ressources

import time

CHANNELS = 4
MIC_ARRAY = load_mic_array_from_ressources("ReSpeaker_USB")
CHUNK_SIZE = 256

OUTPUT_FILE = '/home/rave/RAVE/audio/test'
# Params: .wav file channels, int16 byte size, sampling rate, nb of samples,
#         compression type, compression name
TIME_MAX = 0
FILE_PARAMS = (
    4,
    2,
    CONST.SAMPLING_RATE,
    CONST.SAMPLING_RATE * TIME_MAX,
    "NONE",
    "not compressed",
)
TIME = float('inf') if TIME == 0 else TIME = TIME_MAX

source = MicSource(channels=4, mic_arr=MIC_ARRAY, chunk_size=CHUNK_SIZE)
sink = WavSink(file=OUTPUT_FILE, wav_params=FILE_PARAMS, chunk_size=CHUNK_SIZE)

samples = 0
loop_idx = 0
total_source_time = 0
total_sink_time = 0
while samples / CONST.SAMPLING_RATE < TIME:
    loop_idx += 1

    start_source_time = time.perf_counter_ns()
    x = source()
    if x is None:
        print('End of transmission. Closing.')
        break
    end_source_time = time.perf_counter_ns()

    start_sink_time = time.perf_counter_ns()
    sink(x)
    end_sink_time = time.perf_counter_ns()

    samples += CHUNK_SIZE

    total_source_time += (end_source_time - start_source_time) / 1000000
    total_sink_time += (end_sink_time - start_sink_time) / 1000000


print(f'Mean read time: {total_source_time/loop_idx} ms.')
print(f'Mean send time: {total_sink_time/loop_idx} ms.')
