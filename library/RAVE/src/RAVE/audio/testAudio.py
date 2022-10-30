from pyodas.io import MicSource, WavSink, PlaybackSink
from pyodas.utils import CONST

new_file = "./mic_output.wav"

CHUNK_SIZE = 256
CHANNELS = 8
INT16_BYTE_SIZE = 2
# NB_OF_FRAMES is 0 because we want to record
# for an indefinite amount of time
NB_OF_FRAMES = 0
file_params = (CHANNELS,
               INT16_BYTE_SIZE,
               CONST.SAMPLING_RATE,
               NB_OF_FRAMES,
               'NONE',
               'NONE')

mic_source = MicSource(CHANNELS, chunk_size=CHUNK_SIZE, mic_index=4)
# playback_sink = PlaybackSink(1)
wav_sink = WavSink(new_file, file_params)

# Number of samples processed
samples = 0

# Record for 10 seconds
# while samples/CONST.SAMPLING_RATE < 5:
while True:
    # Get signal from microphone
    x = mic_source()

    # Save signal to a wav file
    wav_sink(x)
    # playback_sink(x[0][None, ...])
    # print(f"Sample: {x}")

    # Increment samples
    samples += CHUNK_SIZE
