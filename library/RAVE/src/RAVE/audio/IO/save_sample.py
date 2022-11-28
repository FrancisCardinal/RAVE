from pyodas.io import WavSource, MicSource, PlaybackSink, WavSink
from pyodas.utils import CONST, load_mic_array_from_ressources

import os
import time
import argparse
import yaml
from tkinter import filedialog

from pathlib import Path

CHANNELS = 4
MIC_ARRAY = load_mic_array_from_ressources("ReSpeaker_USB")
CHUNK_SIZE = 256

OUTPUT_FOLDER = '/home/rave/'
# OUTPUT_FOLDER = 'C:\\GitProjet\\RAVE\\library\\RAVE\\src\\RAVE\\audio\\test_output.wav'

# Params: .wav file channels, int16 byte size, sampling rate, nb of samples,
#         compression type, compression name


# Storing hierarchy: Location -> Speech/Noise -> Name.wav
def generate_output(run_args):

    # Get output path
    loc = run_args.location
    # is_speech = run_args.name.startswith("clsnp") or run_args.name.startswith("p2") or run_args.name.startswith("p3")
    speech_noise = "speech" if args.speech else "noise"
    name = run_args.name
    output_folder_path = os.path.join(OUTPUT_FOLDER, loc, speech_noise, name)
    output_file_path = os.path.join(output_folder_path, 'audio.wav')

    # Save file information
    room_name = os.path.split(loc)[0]
    position = os.path.split(loc)[1]
    output_config_path = os.path.join(output_folder_path, 'configs.yaml')
    config_dict = dict(
        path=output_folder_path,
        room=room_name,
        location=position,
        direction=args.direction,
        is_speech=args.speech,
        sound=name
    )

    Path(output_folder_path).mkdir(parents=True, exist_ok=True)

    with open(output_config_path, "w") as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=None)

    return output_file_path


def main(run_args):

    file_params = (
        8,
        2,
        CONST.SAMPLING_RATE,
        CONST.SAMPLING_RATE * run_args.time,
        "NONE",
        "not compressed",
    )

    output_path = generate_output(run_args)

    source = MicSource(channels=8, chunk_size=CHUNK_SIZE, mic_index=run_args.mic_idx)
    sink = WavSink(file=output_path, wav_params=file_params, chunk_size=CHUNK_SIZE)

    samples = 0
    if run_args.debug:
        loop_idx = 0
        total_source_time = 0
        total_sink_time = 0
    while samples / CONST.SAMPLING_RATE < run_args.time:
        if run_args.debug:
            loop_idx += 1

        if run_args.debug: start_source_time = time.perf_counter_ns()
        x = source()
        if x is None:
            print('End of transmission. Closing.')
            break
        if run_args.debug: end_source_time = time.perf_counter_ns()

        if run_args.debug: start_sink_time = time.perf_counter_ns()
        sink(x)
        if run_args.debug: end_sink_time = time.perf_counter_ns()

        samples += CHUNK_SIZE
        if samples % (CHUNK_SIZE * 25) == 0:
            print(f'Samples processed: {samples}')

        if run_args.debug:
            total_source_time += (end_source_time - start_source_time) / 1000000
            total_sink_time += (end_sink_time - start_sink_time) / 1000000

    print(f'Finished reading {samples} samples by {CHUNK_SIZE} chunk size ')
    if run_args.debug and samples > 0:
        print(f'Mean read time: {total_source_time/loop_idx} ms.')
        print(f'Mean send time: {total_sink_time/loop_idx} ms.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Run the script in debug mode. Is more verbose."
    )
    parser.add_argument(
        "-s", "--speech", action="store_true", help="Sample to record is speech."
    )
    # parser.add_argument(
    #     "-r",
    #     "--room",
    #     action="store",
    #     type=str,
    #     default='tkinter',
    #     choices=['small', 'medium', 'large', 'outside', 'tkinter'],
    #     help="Room in which recordings are done."
    # )
    parser.add_argument(
        "-l",
        "--location",
        action="store",
        type=str,
        default='tkinter',
        help="Room and location used in the room (in the format room/location)."
             "Location format is side, depth, height in cm (x_y_z)."
    )
    parser.add_argument(
        "-n",
        "--name",
        action="store",
        type=str,
        help="Filename to be used for saving."
    )
    parser.add_argument(
        "--direction",
        action="store",
        type=str,
        default='0,0,0',
        help="Direction of the sound compared to the microphone array (side, depth, height in m)."
    )

    parser.add_argument(
        "-t",
        "--time",
        action="store",
        type=int,
        default=10,
        help="Time to save sample for."
    )

    parser.add_argument(
        "-m",
        "--mic_idx",
        action="store",
        type=int,
        default=4,
        help="Microphone index to use for saving sound."
    )

    args = parser.parse_args()

    # # parse room folder
    # room = args.room
    # if room == 'tkinter':
    #     room_path = filedialog.askdirectory(title="Room directory.")
    #     room = os.path.split(room_path)[-1]

    # parse location folder
    if args.location == 'tkinter':
        location_path = filedialog.askdirectory(title="Location directory.")
        split_path = os.path.split(location_path)
        room = split_path[-2]
        location = split_path[-1]
        args.location = os.path.join(room, location)

    main(args)
