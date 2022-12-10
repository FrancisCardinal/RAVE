import argparse
from tkinter import filedialog
import glob
import os
import yaml

from tqdm import tqdm

import torch
import torchaudio
from torchmetrics import SignalNoiseRatio, SignalDistortionRatio

OFFSET = 100


def calculate_SDR(audio_file, debug, snr_sdr_dict):

    original_path = audio_file
    input_dir = os.path.dirname(audio_file)
    target_path = os.path.join(input_dir, 'target.wav')
    output_path = os.path.join(input_dir, 'output.wav')

    # New
    prediction, _ = torchaudio.load(output_path)
    length = prediction.shape[1]
    prediction = prediction[:,OFFSET:length]

    target, _ = torchaudio.load(target_path)
    target = torch.unsqueeze(torch.mean(target, dim=0), dim=0)
    target = target[:, 0:length-OFFSET]

    # Old
    original, _ = torchaudio.load(original_path)
    original = original[:, :length]
    original = torch.mean(original, dim=0, keepdim=True)

    target_original, _ = torchaudio.load(target_path)
    target_original = torch.unsqueeze(torch.mean(target_original, dim=0), dim=0)
    target_original = target_original[:, 0:length]

    sdr = SignalDistortionRatio()
    old_sdr = sdr(original, target_original).item()
    new_sdr = sdr(prediction, target).item()
    gain_sdr = new_sdr - old_sdr
    if debug:
        print("SDR: Before: ", old_sdr, " After: ", new_sdr, " Gain:", gain_sdr)

    snr = SignalNoiseRatio()
    old_snr = snr(original, target_original).item()
    new_snr = snr(prediction, target).item()
    gain_snr = new_snr - old_snr
    if debug:
        print("SNR: Before: ", old_snr, " After: ", new_snr, " Gain: ", gain_snr)

    # Get file configs
    file_configs_path = os.path.join(input_dir, 'configs.yaml')
    with open(file_configs_path, "r") as stream:
        try:
            file_configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("Couldn't load audio file configs.")
            print(exc)
            exit()

    # Add values to configs and save
    file_configs['old_snr'] = old_snr
    file_configs['new_snr'] = new_snr
    file_configs['gain_snr'] = gain_snr
    file_configs['old_sdr'] = old_sdr
    file_configs['new_sdr'] = new_sdr
    file_configs['gain_sdr'] = gain_sdr
    with open(file_configs_path, "w") as outfile:
        yaml.dump(file_configs, outfile, default_flow_style=None)

    # Update mean values
    n = snr_sdr_dict['count']
    snr_sdr_dict['old_snr'] = snr_sdr_dict['old_snr'] * (n - 1) / n + old_snr / n
    snr_sdr_dict['new_snr'] = snr_sdr_dict['new_snr'] * (n - 1) / n + new_snr / n
    snr_sdr_dict['old_sdr'] = snr_sdr_dict['old_sdr'] * (n - 1) / n + old_sdr / n
    snr_sdr_dict['new_sdr'] = snr_sdr_dict['new_sdr'] * (n - 1) / n + new_sdr / n

    if debug:
        print(f"Finished calculating SDR on {audio_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Run in debug mode"
    )
    parser.add_argument(
        "-s",
        "--sources",
        action="store",
        type=str,
        default='tkinter',
        help="Absolute path to audio sources to enhance",
    )
    args = parser.parse_args()

    # Parse paths
    source_subfolder = args.sources
    if source_subfolder == 'tkinter':
        source_subfolder = filedialog.askdirectory(title="Audio extracts dataset folder")

    # Get all files
    audio_files = glob.glob(os.path.join(source_subfolder, '**', 'audio.wav'), recursive=True)

    # Create output dict
    snr_sdr_dict = dict(
        count=0,
        old_snr=0,
        new_snr=0,
        gain_snr=0,
        old_sdr=0,
        gain_sdr=0,
        new_sdr=0
    )

    # TODO: multiprocess
    for audio_file in tqdm(audio_files, desc="Calculating SDR"):
        snr_sdr_dict['count'] += 1
        calculate_SDR(audio_file, args.debug, snr_sdr_dict)

    # Calculate gain
    snr_sdr_dict['gain_snr'] = snr_sdr_dict['new_snr'] - snr_sdr_dict['old_snr']
    snr_sdr_dict['gain_sdr'] = snr_sdr_dict['new_sdr'] - snr_sdr_dict['old_sdr']

    # Print results in yaml file at source directory
    results_yaml_path = os.path.join(source_subfolder, 'sdr_results.yaml')
    with open(results_yaml_path, "w") as outfile:
        yaml.dump(snr_sdr_dict, outfile, default_flow_style=None)

    print(f"SDR_SNR: Measured over {snr_sdr_dict['count']} samples :")
    print(f"\t Old SNR: {snr_sdr_dict['old_snr']} \t New SNR: {snr_sdr_dict['new_snr']} "
          f"\t Gain SNR: {snr_sdr_dict['gain_snr']}")
    print(f"\t Old SDR: {snr_sdr_dict['old_sdr']} \t New SDR: {snr_sdr_dict['new_sdr']}"
          f"\t Gain SDR: {snr_sdr_dict['gain_sdr']}")

