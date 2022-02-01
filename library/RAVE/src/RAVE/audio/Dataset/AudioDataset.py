import torch

from tkinter import filedialog
import glob
import os
import random
import yaml
import soundfile as sf
import numpy as np

from audio import AudioDatasetBuilder

MINIMUM_DATASET_LENGTH = 500

class AudioDataset(torch.utils.data.Dataset):
    """
    Class handling usage of the audio dataset
    """

    def __init__(
        self,
        dataset_path=None,
        data_split=None,
        generate_dataset_runtime=False,
        is_debug=False
    ):
        self.is_debug = is_debug

        # Load params/configs
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_config.yaml')
        with open(config_path, "r") as stream:
            try:
                self.configs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # If need generator, get item
        self.dataset_builder = AudioDatasetBuilder(self.configs['source_folder'],
                                                   self.configs['noise_folder'],
                                                   dataset_path,
                                                   self.configs['noise_count_range'],
                                                   self.configs['speech_as_noise'],
                                                   self.configs['sample_per_speech'],
                                                   self.is_debug)

        # Set up dataset (if no folder, open tkinter to choose)
        self.dataset_path = dataset_path
        if not self.dataset_path:
            self.dataset_path = filedialog.askdirectory(title="Choose a dataset folder")
        self.data = []
        self.get_dataset(self.dataset_path)

        self.generate_dataset_runtime = generate_dataset_runtime

        # # Split dataset into subsets
        # if not data_split:
        #     data_split = self.configs['data_split']
        # random.shuffle()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get chosen items
        if self.generate_dataset_runtime:
            audio_signal, audio_mask, speech_mask, noise_mask, config_dict = self.run_dataset_builder()
        else:
            item_path = self.data[idx]
            audio_signal, audio_mask, speech_mask, noise_mask, config_dict = self.load_item_from_disk(item_path)

        # Format
        # TODO: Cut segments into correct length

        return audio_signal, speech_mask, config_dict

    def get_dataset(self, dataset_path):

        # TODO: Other way to check if dataset is ok?
        # Check if exists, and if contains files
        if os.path.isdir(dataset_path) and os.listdir(dataset_path):
            # If contains files, check if files are audio.wav (dataset files)
            wav_file_list = glob.glob(os.path.join(dataset_path, '**', 'audio.wav'))
            if len(wav_file_list) == 0:
                print(f"ERROR: DATASET: Found files in folder ({dataset_path}) but none pertaining to dataset. "
                      f"Exiting")
                exit()
            elif len(wav_file_list) < MINIMUM_DATASET_LENGTH and self.is_debug:
                print(f"ERROR: DATASET: Found files in dataset ({dataset_path}), but only {len(wav_file_list)} "
                      f"(under minimum limit {MINIMUM_DATASET_LENGTH}). Exiting")
                exit()
            else:
                # Load dataset paths
                for wav_file in wav_file_list:
                    self.data.append(os.path.dirname(wav_file))
                return

        # If didn't exit or find dataset, create dataset
        print(f"DATASET: No dataset found at '{dataset_path}', generating new one.")
        file_count, dataset_list = self.dataset_builder.generate_dataset(save_run=True)
        print(f"DATASET: Dataset generated at '{dataset_path}'. {file_count} files generated")
        self.data = dataset_list

    def load_item_from_disk(self, subfolder_path):

        # Get paths for files
        audio_file_path = os.path.join(subfolder_path, 'audio.wav')
        audio_mask_path = os.path.join(subfolder_path, 'combined_audio_gt.npz')
        speech_mask_path = os.path.join(subfolder_path, 'source.npz')
        noise_mask_path = os.path.join(subfolder_path, 'noise.npz')
        configs_file_path = os.path.join(subfolder_path, 'configs.yaml')

        # Get config dict
        with open(configs_file_path, "r") as stream:
            try:
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # Get audio file
        audio_signal, fs = sf.read(audio_file_path)
        audio_mask = np.load(audio_mask_path)
        speech_mask = np.load(speech_mask_path)
        noise_mask = np.load(noise_mask_path)

        return audio_signal, audio_mask, speech_mask, noise_mask, config_dict

    def run_dataset_builder(self):

        # Get generator need arguments
        rooms = self.configs['room']
        room_size = rooms[np.random.randint(0, len(rooms))]

        results = self.dataset_builder.generate_single_run(room_size)

        audio_signal = results['audio']
        audio_mask = results['combined_audio_gt']
        speech_mask = results['source']
        noise_mask = results['noise']
        config_dict = results['configs']

        return audio_signal, audio_mask, speech_mask, noise_mask, config_dict