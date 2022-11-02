import pyodas.core as core
import numpy as np
from pyodas.io import MicSource, WavSink  # PlaybackSink
from pyodas.utils import CONST, get_delays_based_on_mic_array

new_file = "./mic_output.wav"

CHUNK_SIZE = 256
FRAME_SIZE = 2 * CHUNK_SIZE
CHANNELS = 8
INT16_BYTE_SIZE = 2
# NB_OF_FRAMES is 0 because we want to record
# for an indefinite amount of time
NB_OF_FRAMES = 0
file_params = (CHANNELS, INT16_BYTE_SIZE, CONST.SAMPLING_RATE, NB_OF_FRAMES, "NONE", "NONE")
file_params_mono = (1, INT16_BYTE_SIZE, CONST.SAMPLING_RATE, NB_OF_FRAMES, "NONE", "NONE")
mic_dict = {
    "mics": {
        "0": [-0.07055, 0, 0],
        "1": [-0.07055, 0.0381, 0],
        "2": [-0.05715, 0.0618, 0],
        "3": [-0.01905, 0.0618, 0],
        "4": [0.01905, 0.0618, 0],
        "5": [0.05715, 0.0618, 0],
        "6": [0.07055, 0.0381, 0],
        "7": [0.07055, 0, 0],
    },
    "nb_of_channels": 8,
}

mic_source = MicSource(CHANNELS, mic_arr=mic_dict, chunk_size=CHUNK_SIZE, mic_index=5)
# playback_sink = PlaybackSink(1)
wav_sink = WavSink(new_file, file_params)
wav_sink_ds = WavSink("./ds_output", file_params_mono)

stft = core.Stft(CHANNELS, FRAME_SIZE, "sqrt_hann")
DS = core.DelaySum(FRAME_SIZE)
istft = core.IStft(1, FRAME_SIZE, CHUNK_SIZE, "sqrt_hann")

target = np.array([[0, 1, 0.5]])
delay = get_delays_based_on_mic_array(target, frame_size=FRAME_SIZE, mic_array=mic_source.mic_array)[0]

# Number of samples processed
samples = 0

# Record for 10 seconds
while samples / CONST.SAMPLING_RATE < 5:
    # while True:
    # Get signal from microphone
    x = mic_source()
    wav_sink(x)

    X = stft(x)
    Y = DS(X, delay)
    y = istft(Y)

    # Save signal to a wav file
    wav_sink_ds(y)
    # playback_sink(x[0][None, ...])
    # print(f"Sample: {x}")

    # Increment samples
    samples += CHUNK_SIZE
