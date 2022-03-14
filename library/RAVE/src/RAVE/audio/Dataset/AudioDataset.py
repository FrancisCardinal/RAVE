import torch
import torchaudio
import torchvision
from pyodas.utils import generate_mic_array, get_delays_based_on_mic_array
from pyodas.core import IStft

from tkinter import filedialog
import glob
import os
import random
import yaml
import soundfile as sf
import numpy as np
import math

from .AudioDatasetBuilder import AudioDatasetBuilder
from RAVE.audio.Beamformer.Beamformer import Beamformer

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
        num_samples = 32000
    ):
        self.duration = num_samples / sample_rate
        self.is_debug = IS_DEBUG
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.hop_len = 256
        self.nb_chunks = math.floor((num_samples / self.hop_len) + 1)
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
            n_fft=1024,
            hop_length= self.hop_len,
            power=None,
            return_complex=True,
            normalized=True
        )
        self.waveform = torchaudio.transforms.InverseSpectrogram(
             n_fft=1024,
            hop_length= 256
        )
        self.delay_and_sum = Beamformer('delaysum', frame_size=1024)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """if self.generate_dataset_runtime:
            audio_signal, audio_mask, speech_mask, noise_mask, config_dict = self.run_dataset_builder() # todo: audio_signal needs to be a tensor 
        else:"""
        item_path = self.data[idx]
        audio_signal, audio_sr, noise_target, noise_sr, speech_target, speech_sr, config_dict = self.load_item_from_disk(item_path)

        min_val = min(audio_signal.shape[1], noise_target.shape[1], speech_target.shape[1])

        begin = random.randint(0, (min_val - self.num_samples)) if min_val > self.num_samples + 1 else 0

        signal1 = self._formatAndConvertToSpectogram(audio_signal, audio_sr, begin)
        signal2 = self._delaySum(audio_signal, audio_sr, config_dict, begin)
        noise_target = self._formatAndConvertToSpectogram(noise_target, noise_sr, begin)
        speech_target = self._formatAndConvertToSpectogram(speech_target, speech_sr, begin)

        signal = torch.cat([signal1, signal2], dim=1)

        target = noise_target**2 / (noise_target**2 + speech_target**2 + 1e-10)

        total_energy =noise_target**2 + speech_target**2
        total_energy -= torch.min(total_energy)
        total_energy /= torch.max(total_energy)

        return signal, torch.squeeze(target), total_energy
    
    def _delaySum(self, raw_signal, sr, config, begin):
        signal = self._resample(raw_signal, sr)
        signal = self._cut(signal, begin)
        signal = self._right_pad(signal)
        freq_signal = self.transformation(signal)
        mic0 = list(np.subtract(config['microphones'][0], config['user_pos']))
        mic1 = list(np.subtract(config['microphones'][1], config['user_pos']))
        mic2 = list(np.subtract(config['microphones'][2], config['user_pos']))
        mic3 = list(np.subtract(config['microphones'][3], config['user_pos']))
        
        mic_array =  generate_mic_array({
            "mics": {
                "0": mic0,
                "1": mic1,
                "2": mic2,
                "3": mic3
            },
            "nb_of_channels": 4
        })

        X = freq_signal.cpu().detach().numpy()
        X = np.einsum("ijk->kij", X)
        delay = np.squeeze(get_delays_based_on_mic_array(np.array([config['source_dir']]), mic_array, 1024)) #set variable 1024
        
        signal = np.zeros((1, X.shape[2], X.shape[0]),dtype=complex)
        for index, item in enumerate(X):
            signal[...,index] = self.delay_and_sum(freq_signal=item, delay=delay)

        signal = torch.from_numpy(signal)
        signal = 20*torch.log10(torch.abs(signal) + 1e-09)
        return signal

    def _formatAndConvertToSpectogram(self, raw_signal, sr, begin):
        signal = self._resample(raw_signal, sr)
        signal = self._cut(signal, begin)
        signal = self._right_pad(signal)
        freq_signal = self.transformation(signal)
        freq_signal = freq_signal**2
        freq_signal = self._set_mono(freq_signal)
        freq_signal = 20*torch.log10(torch.abs(freq_signal) + 1e-09)
        return freq_signal

    """def _formatAndConvertToSpectogram(self, raw_signal, sr):
        signal = self._resample(raw_signal, sr)
        signal = self._cut(signal)
        signal = self._right_pad(signal)

        signal = self._set_mono(signal)
        freq_signal = self.transformation(signal)
        freq_signal = 20*torch.log10(torch.abs(freq_signal) + 1e-09)
        return freq_signal"""

    def _resample(self, signal, sr):
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            signal =resampler(signal)
        return signal

    @staticmethod
    def _set_mono(signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut(self, signal, begin):
        #begin = 0
        if signal.shape[1] > self.num_samples:
            signal = signal[:, begin: begin+self.num_samples]
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
        dataset_list = np.array([str(i) for i in dataset_list], dtype=np.str)
        self.data = dataset_list

    @staticmethod
    def load_item_from_disk(subfolder_path):

        # Get paths for files
        audio_file_path = os.path.join(subfolder_path, 'audio.wav')
        noise_target_path = os.path.join(subfolder_path, 'noise.wav')
        speech_target_path = os.path.join(subfolder_path, 'speech.wav')
        configs_file_path = os.path.join(subfolder_path, 'configs.yaml')

        # Get config dict
        with open(configs_file_path, "r") as stream:
            try:
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # Get audio file
        audio_signal, audio_sr = torchaudio.load(audio_file_path)
        noise_target, noise_sr = torchaudio.load(noise_target_path)
        speech_target, speech_sr = torchaudio.load(speech_target_path)

        return audio_signal, audio_sr, noise_target, noise_sr, speech_target, speech_sr, config_dict

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