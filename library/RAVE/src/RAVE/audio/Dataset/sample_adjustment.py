# Imports
import argparse
from tkinter import filedialog
import os
import glob
import yaml

# import numpy as np
# import math
# from scipy.io import wavfile
# import scipy.signal as sps
# import soundfile as sf

from pydub import AudioSegment, effects


def save_data(data, run_args): #, global_mean):
    # Create folder
    output_dir = os.path.normpath(os.path.join(run_args.output, data['configs']['path']))
    os.makedirs(output_dir, exist_ok=True)

    # Save data files
    data_path = os.path.join(output_dir, 'audio.wav')
    data['downsampled_data'].export(data_path, format="wav")
    if run_args.debug:
        # Save original data
        og_data_path = os.path.join(output_dir, 'original.wav')
        data['data'].export(og_data_path, format="wav")
        # Save upsampled normalized
        norm_data_path = os.path.join(output_dir, 'normalized.wav')
        data['normalized_data'].export(norm_data_path, format="wav")
        # Save low_passed
        lowpass_data_path = os.path.join(output_dir, 'lowpass.wav')
        data['lowpassed_data'].export(lowpass_data_path, format="wav")

    # Save configs
    configs = data['configs']
    configs['sampling_rate'] = run_args.rate
    config_dict_file_name = os.path.join(output_dir, "configs.yaml")
    with open(config_dict_file_name, "w") as outfile:
        yaml.dump(configs, outfile, default_flow_style=None)


def main(run_args):
    # Get every .wav and config files
    wav_file_paths = glob.glob(os.path.join(run_args.source, '**', "*.wav"), recursive=True)
    dir_path_list = [os.path.dirname(i) for i in wav_file_paths]

    # Get audio path, data, downsample, get mean amplitude
    for dir_path in dir_path_list:
        wav_dict = dict()

        # Get configs
        config_path = os.path.join(dir_path, 'configs.yaml')
        with open(config_path, "r") as stream:
            try:
                configs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        wav_dict['configs'] = configs

        # Get audio
        audio_segment = AudioSegment.from_wav(os.path.join(dir_path, 'audio.wav'))
        wav_dict['data'] = audio_segment
        # Lowpass
        lowpassed_audio_segment = audio_segment.low_pass_filter(2000)
        wav_dict['lowpassed_data'] = lowpassed_audio_segment
        # Normalize
        normalized_audio_segment = effects.normalize(lowpassed_audio_segment)
        wav_dict['normalized_data'] = normalized_audio_segment
        # Downsample
        downsampled_audio_segment = normalized_audio_segment.set_frame_rate(run_args.rate)
        wav_dict['downsampled_data'] = downsampled_audio_segment

        save_data(wav_dict, run_args)

        if run_args.debug:
            print(f'Finished processing sample at {dir_path}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Run the script in debug mode. Is more verbose."
    )
    parser.add_argument(
        "-r",
        "--rate",
        action="store",
        type=int,
        default=16000,
        help="New sampling rate."
    )
    parser.add_argument(
        "-s",
        "--source",
        action="store",
        type=str,
        default='tkinter',
        help="Source folder containing all .wav files to modify."
    )
    parser.add_argument(
        "-o",
        "--output",
        action="store",
        type=str,
        default='tkinter',
        help="Output folder to drop modified data."
    )

    args = parser.parse_args()

    # parse folders
    if args.source == 'tkinter':
        args.source = filedialog.askdirectory(title="Source directory.")
    if args.output == 'tkinter':
        args.output = filedialog.askdirectory(title="Output directory.")

    main(args)
