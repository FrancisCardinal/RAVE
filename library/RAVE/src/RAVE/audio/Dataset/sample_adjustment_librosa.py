# Imports
import argparse
from tkinter import filedialog
import os
import glob

import numpy as np
import yaml

import librosa
import soundfile as sf


def save_data(data, run_args, old_rate):
    # Create folder
    output_dir = os.path.normpath(os.path.join(run_args.output, data['configs']['path']))
    os.makedirs(output_dir, exist_ok=True)

    # Save data files
    data_path = os.path.join(output_dir, 'audio.wav')
    sf.write(data_path,  data['downsampled_data'].T, run_args.rate, 'PCM_16')
    if run_args.debug:
        # Save original data
        og_data_path = os.path.join(output_dir, 'original.wav')
        sf.write(og_data_path, data['data'].T, old_rate, 'PCM_16')
        # Save upsampled normalized
        if run_args.normalize:
            norm_data_path = os.path.join(output_dir, 'normalized.wav')
            sf.write(norm_data_path, data['normalized_data'].T, old_rate, 'PCM_16')

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
        data, samplerate = sf.read(os.path.join(dir_path, 'audio.wav'), dtype='float32')
        data = data.T
        wav_dict['data'] = data

        # Normalize
        data_norm = data
        if run_args.normalize:
            max_val = np.amax(data)
            min_val = np.abs(np.amin(data))
            factor = max(max_val, min_val)
            data_norm = data / factor
            wav_dict['normalized_data'] = data_norm
            # Check norm
            if run_args.test:
                data_min = []
                data_max = []
                norm_min = []
                norm_max = []
                for i in range(len(data)):
                    data_min.append(np.amin(data[i]))
                    data_max.append(np.amax(data[i]))
                    norm_min.append(np.amin(data_norm[i]))
                    norm_max.append(np.amax(data_norm[i]))

        # Downsample
        data_16k = librosa.resample(data_norm, orig_sr=samplerate, target_sr=run_args.rate, res_type="soxr_vhq")
        wav_dict['downsampled_data'] = data_16k

        save_data(wav_dict, run_args, samplerate)

        if run_args.debug:
            print(f'Finished processing sample at {dir_path}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Run the script in debug mode. Is more verbose."
    )
    parser.add_argument(
        "-t", "--test", action="store_true", help="Run the script in test mode. Tests normalisation."
    )
    parser.add_argument(
        "-n", "--normalize", action="store_true", help="Normalize sample amplitude (volume)."
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
