import numpy as np
import argparse
from tkinter import filedialog
import torch
import os
import yaml
import time
import glob

from pyodas.core import (
    Stft,
    IStft,
    KissMask,
    SpatialCov,
    DelaySum
)
from pyodas.utils import CONST, generate_mic_array, load_mic_array_from_ressources, get_delays_based_on_mic_array

from RAVE.audio.IO.IO_manager import IOManager
from RAVE.audio.Neural_Network.AudioModel import AudioModel
from RAVE.audio.Beamformer.Beamformer import Beamformer
from RAVE.audio.AudioManager import AudioManager

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

# Constants
# TODO: Uniformise microphone array in config file or such
# MIC_ARRAY = generate_mic_array({
#     'mics': {
#         '0': [-0.1, -0.1, 0],
#         '1': [-0.1, 0.1, 0],
#         '2': [0.1, -0.1, 0],
#         '3': [0.1, 0.1, 0]
#     },
#     'nb_of_channels': 4
# })
MIC_DICT = load_mic_array_from_ressources('ReSpeaker_USB')
MIC_ARRAY = generate_mic_array(load_mic_array_from_ressources('ReSpeaker_USB'))
CHANNELS = 4
OUT_CHANNELS = 4

CHUNK_SIZE = 256
FRAME_SIZE = 4 * CHUNK_SIZE

DEFAULT_OUTPUT_FOLDER = 'C:\GitProjet\pipeline'
BEST_MODEL_PATH = 'C:\\GitProjet\\RAVE\\library\\RAVE\\src\\RAVE\\audio\\Neural_Network\\model\\saved_model.pth'

EPSILON = 1e-9

def main(DEBUG, INPUT, OUTPUT, TIME, MASK):
    # TODO: add multiple source/sink support?
    # TODO: simplify io abstraction kwargs

    # Params: .wav file channels, int16 byte size, sampling rate, nb of samples,
    #         compression type, compression name
    FILE_PARAMS = (
        4,
        2,
        CONST.SAMPLING_RATE,
        CONST.SAMPLING_RATE * TIME,
        "NONE",
        "not compressed",
    )
    mic_array = MIC_ARRAY
    channels = FILE_PARAMS[0]
    target = np.array([0, 1, 0.5])
    subfolder_path = None
    original_sink = None

    # IO
    io_manager = IOManager()
    # Input source
    if INPUT == 'default' or INPUT.isdigit():
        # MicSource
        if INPUT == 'default':
            INPUT = None
        else:
            INPUT = int(INPUT)
        source = io_manager.add_source(source_name='MicSource1', source_type='mic', mic_index=INPUT,
                                       channels=CHANNELS, mic_arr=MIC_DICT, chunk_size=CHUNK_SIZE)
    else:
        # WavSource
        source = io_manager.add_source(source_name='WavSource1', source_type='sim',
                                       file=INPUT, chunk_size=CHUNK_SIZE)
        FILE_PARAMS = source.wav_params

        # If .wav source, get more info from files
        # Paths
        # TODO: fix output to .wav and playback?
        # TODO: fix folder path when no .wav input?
        subfolder_path = os.path.split(INPUT)[0]
        config_path = os.path.join(subfolder_path, 'configs.yaml')
        with open(config_path, "r") as stream:
            try:
                configs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                exit()
        target = configs['source_dir']

        # Microphone array
        mic_dict = {'mics': dict(), 'nb_of_channels': 0}
        for mic_idx, mic in enumerate(configs['mic_rel']):
            mic_dict['mics'][f'{mic_idx}'] = mic
            mic_dict['nb_of_channels'] += 1
        mic_array = generate_mic_array(mic_dict)
        channels = mic_dict['nb_of_channels']

    # Output sink
    if OUTPUT == 'default' or OUTPUT.isdigit():
        # PlaybackSink
        if OUTPUT == 'default':
            OUTPUT = None
        else:
            OUTPUT = int(OUTPUT)
        output_sink = io_manager.add_sink(sink_name='original', sink_type='play', device_index=OUTPUT,
                                          channels=OUT_CHANNELS, chunk_size=CHUNK_SIZE)
    else:
        # If mic source (no subfolder), output subfolder
        if subfolder_path == None:
            loop_idx = 0
            subfolder_path = os.path.join(DEFAULT_OUTPUT_FOLDER, 'run')
            while os.path.exists(f'{subfolder_path}_{loop_idx}'):
                loop_idx += 1
            os.makedirs(subfolder_path)

        # WavSink
        original = os.path.join(subfolder_path, 'original.wav')
        original_sink = io_manager.add_sink(sink_name='original', sink_type='sim',
                                            file=original, wav_params=FILE_PARAMS, chunk_size=CHUNK_SIZE)
        if OUTPUT == '':
            OUTPUT = os.path.join(subfolder_path, 'output_noise.wav')
        output_sink = io_manager.add_sink(sink_name='output', sink_type='sim',
                                          file=OUTPUT, wav_params=FILE_PARAMS, chunk_size=CHUNK_SIZE)

    # Masks
    # TODO: Add abstraction to model for info not needed in main script
    if MASK:
        masks = KissMask(mic_array, buffer_size=30)
    else:
        model = AudioModel(input_size=1026, hidden_size=256, num_layers=2)
        model.to(DEVICE)
        if DEBUG:
            print(model)
        model.load_best_model(BEST_MODEL_PATH, DEVICE)
        delay_and_sum = DelaySum(FRAME_SIZE)

    # Beamformer
    # TODO: simplify beamformer abstraction kwargs
    beamformer = Beamformer(name='mvdr', channels=channels)

    # Utils
    # TODO: Check if we want to handle stft and istft in IOManager class
    stft = Stft(channels, FRAME_SIZE, "hann")
    istft = IStft(channels, FRAME_SIZE, CHUNK_SIZE, "hann")
    speech_spatial_cov = SpatialCov(channels, FRAME_SIZE, weight=0.5)
    noise_spatial_cov = SpatialCov(channels, FRAME_SIZE, weight=0.5)

    # Record for 6 seconds
    samples = 0
    loop_idx = 0
    total_time = 0
    total_beamformer_time = 0
    total_mask_time = 0
    total_network_time = 0
    total_data_time = 0
    max_time = 0
    while samples / CONST.SAMPLING_RATE < TIME:
        # Start time
        loop_idx += 1
        start_time = time.perf_counter_ns()

        # Record from microphone
        x = source()
        if x is None:
            print('End of transmission. Closing.')
            break

        # Save the unprocessed recording
        if original_sink:
            original_sink(x)
        X = stft(x)

        # Compute the masks
        if MASK:
            start_mask_time = time.perf_counter_ns()
            speech_mask, noise_mask = masks(X, target)
            end_mask_time = time.perf_counter_ns()
            total_mask_time += (end_mask_time - start_mask_time) / 1000000
        else:
            start_data_time = time.perf_counter_ns()
            # Delay and sum
            target_np = np.array([target])
            delay = get_delays_based_on_mic_array(target_np, MIC_ARRAY, FRAME_SIZE)
            sum = delay_and_sum(X, delay[0])
            sum_tensor = torch.from_numpy(sum)
            sum_db = 20*torch.log10(torch.abs(sum_tensor) + EPSILON)

            # Mono
            energy = torch.from_numpy(X**2)
            mono_X = torch.mean(energy, dim=0, keepdim=True)
            mono_db = 20*torch.log10(torch.abs(mono_X) + EPSILON)

            concat_spec = torch.cat([mono_db, sum_db], dim=1)
            concat_spec = torch.reshape(concat_spec, (1, 1, concat_spec.shape[1], 1))
            start_network_time = time.perf_counter_ns()
            with torch.no_grad():
                noise_mask = model(concat_spec)
            end_network_time = time.perf_counter_ns()
            noise_mask = torch.squeeze(noise_mask).numpy()
            speech_mask = 1 - noise_mask
            total_network_time += (end_network_time - start_network_time) / 1000000
            total_data_time += ((end_network_time - start_data_time) - (end_network_time - start_network_time)) / 1000000

        # Spatial covariance matrices
        target_scm = speech_spatial_cov(X, speech_mask)
        noise_scm = noise_spatial_cov(X, noise_mask)

        # MVDR
        start_bf_time = time.perf_counter_ns()
        Y = beamformer(signal=X, target_scm=target_scm, noise_scm=noise_scm)
        end_bf_time = time.perf_counter_ns()
        total_beamformer_time += (end_bf_time - start_bf_time) / 1000000
        y = istft(Y)
        out = y
        if OUT_CHANNELS == 1:
            channel_to_use = y.shape[0]//2
            out = y[channel_to_use]
        elif OUT_CHANNELS == 2:
            out = np.array([y[0], y[-1]])
        output_sink(out)

        samples += CHUNK_SIZE
        end_time = time.perf_counter_ns()
        loop_time = (end_time - start_time) / 1000000
        total_time += loop_time
        max_time = loop_time if loop_time > max_time else max_time

        if DEBUG and samples % (CHUNK_SIZE*50) == 0:
            print(f'Samples processed: {samples}')
            print(f'Time for loop: {loop_time} ms.')

    print('Finished running main_audio')
    print(f'Mean time per loop: {total_time / loop_idx} ms.')
    print(f'Longest time per loop: {max_time} ms.')
    print(f'Mean beamformer time per loop: {total_beamformer_time / loop_idx} ms.')
    if MASK:
        print(f'Mean mask time per loop: {total_mask_time / loop_idx} ms.')
    else:
        print(f'Mean data time per loop: {total_data_time / loop_idx} ms.')
        print(f'Mean network per loop: {total_network_time / loop_idx} ms.')


def main2(LOOP_DIR=None):

    # Get all files in a subdirectory
    if LOOP_DIR:
        audio_manager = AudioManager(debug=False, mask=True, use_timers=False)
        input_files = glob.glob(os.path.join(LOOP_DIR, '**/audio.wav'))
        for audio_file in input_files:
            print(f'Starting speech enhancement on {audio_file}')
            audio_dict = {
                'name': 'loop_sim_source',
                'type': 'sim',
                'file': audio_file
            }
            audio_manager.initialise_audio(source=audio_dict)
            audio_manager.main_loop()
    else:
        audio_manager = AudioManager(debug=True, mask=False, use_timers=False)
        audio_manager.initialise_audio()
        audio_manager.main_loop()


if __name__ == "__main__":
    main2()
    # main2(LOOP_DIR='C:\\GitProjet\\MS-SNSD\\output\\2')

    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument(
    #     "-d", "--debug", action="store_true", help="Run the script in debug mode. Is more verbose."
    # )
    # parser.add_argument(
    #     "-m", "--mask", action="store_true", help="Use predefined masks instead of network."
    # )
    #
    # parser.add_argument(
    #     "-i",
    #     "--input",
    #     action="store",
    #     type=str,
    #     default='tkinter',
    #     help="Source to use as input signals (can be microphone matrix or simulated .wav file)."
    # )
    # parser.add_argument(
    #     "-o",
    #     "--output",
    #     action="store",
    #     type=str,
    #     default='',
    #     help="Sink to use to output signals (can be speakers or simulated .wav file)."
    # )
    #
    # parser.add_argument(
    #     "-t",
    #     "--time",
    #     action="store",
    #     type=int,
    #     default=float('inf'),
    #     help="Time to run audio for."
    # )
    # args = parser.parse_args()
    #
    # # parse sources
    # input = args.input
    # if input == 'tkinter':
    #     input = filedialog.askopenfilename(title="Input .wave file.")
    #
    # # parse noises
    # output = args.output
    # if output == 'tkinter':
    #     output = filedialog.askopenfilename(title="Output .wav file.")
    #
    # main(
    #     args.debug,
    #     input,
    #     output,
    #     args.time,
    #     args.mask
    # )
