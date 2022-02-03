import torch
import torchaudio

from tkinter import filedialog
import glob
import os
import random
import yaml
import soundfile as sf
import numpy as np

from .AudioDatasetBuilder import AudioDatasetBuilder

MINIMUM_DATASET_LENGTH = 500
IS_DEBUG = True

class AudioDataset(torch.utils.data.Dataset):
    """
    Class handling usage of the audio dataset
    """

    def __init__(
        self,
        dataset_path=None,
        data_split=None,
        generate_dataset_runtime=False,
        is_debug=False,
        sample_rate = 16000,
        num_samples = 16000,
    ):
        self.is_debug = IS_DEBUG
        self.sample_rate = sample_rate
        self.num_samples = num_samples

        # Load params/configs
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_config.yaml')
        with open(config_path, "r") as stream:
            try:
                self.configs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        

        # Set up dataset (if no folder, open tkinter to choose)
        self.dataset_path = dataset_path
        if not self.dataset_path:
            self.dataset_path = filedialog.askdirectory(title="Choose a dataset folder")
        self.data = []

        # If need generator, get item
        self.dataset_builder = AudioDatasetBuilder(self.configs['source_folder'],
                                                   self.configs['noise_folder'],
                                                   self.dataset_path,
                                                   self.configs['noise_count_range'],
                                                   self.configs['speech_as_noise'],
                                                   self.configs['sample_per_speech'],
                                                   self.is_debug)
        self.get_dataset(self.dataset_path)

        self.generate_dataset_runtime = generate_dataset_runtime

        # # Split dataset into subsets
        # if not data_split:
        #     data_split = self.configs['data_split']
        # random.shuffle()

        self.transformation = torchaudio.transforms.Spectrogram(
            n_fft=1024
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get chosen items
        if self.generate_dataset_runtime:
            audio_signal, audio_mask, speech_mask, noise_mask, config_dict = self.run_dataset_builder() # todo: audio_signal needs to be a tensor 
        else:
            item_path = self.data[idx]
            audio_signal, sr, audio_mask, speech_mask, noise_mask, config_dict = self.load_item_from_disk(item_path)

        # Format
        signal = self._resample(audio_signal, sr)
        signal = self._cut(signal)
        signal = self._right_pad(signal)

        # get the log mean spectogram of the array of mics
        signal1 = self._set_mono(audio_signal)
        signal1 = self.transformation(signal1)
        signal1 = torchaudio.functional.amplitude_to_DB(signal1, multiplier=20.0, amin=1e-10, db_multiplier=1.0 )

        # the spectogram of delay and sum signal

        signal = signal1
        return signal, speech_mask

    def _resample(self, signal, sr):
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            signal =resampler(signal)
        return signal
    
    def _set_mono(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut(self, signal):
        if signal.shape[1] > self.num_samples:
            signal[:, self.num_samples]
        return signal
    
    def _right_pad(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            signal = torch.nn.functional.pad(signal, (0, num_missing_samples))
        return signal

    def get_dataset(self, dataset_path):

        # TODO: Other way to check if dataset is ok?
        # Check if exists, and if contains files
        if os.path.isdir(dataset_path) and os.listdir(dataset_path):
            # If contains files, check if files are audio.wav (dataset files)

            wav_file_list = glob.glob(os.path.join(dataset_path, '**', 'audio.wav'), recursive=True)
            if len(wav_file_list) == 0:
                print(f"ERROR: DATASET: Found files in folder ({dataset_path}) but none pertaining to dataset. "
                      f"Exiting")
                exit()
            elif len(wav_file_list) < MINIMUM_DATASET_LENGTH and not self.is_debug:
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
        #audio_signal, fs = sf.read(audio_file_path)
        audio_signal, sr = torchaudio.load(audio_file_path)
        audio_mask = np.load(audio_mask_path)
        speech_mask = np.load(speech_mask_path)
        noise_mask = np.load(noise_mask_path)

        return audio_signal, sr, audio_mask, speech_mask, noise_mask, config_dict

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