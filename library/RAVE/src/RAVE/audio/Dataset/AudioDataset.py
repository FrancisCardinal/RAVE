import torch
import torchaudio
from pyodas.utils import generate_mic_array, get_delays_based_on_mic_array
from pyodas.core import IStft

from tkinter import filedialog
import glob
import os
import random
import yaml
import soundfile as sf
import numpy as np

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
        num_samples = 48000,
        device= 'cpu'
    ):
        self.device = device
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
            n_fft=1024,
            hop_length= 256,
            power=None,
            return_complex=True
        )
        self.waveform = torchaudio.transforms.InverseSpectrogram(
             n_fft=1024,
            hop_length= 256
        )
        self.delay_and_sum = Beamformer('delaysum', frame_size=1024)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get chosen items
        if self.generate_dataset_runtime:
            audio_signal, audio_mask, speech_mask, noise_mask, config_dict = self.run_dataset_builder() # todo: audio_signal needs to be a tensor 
        else:
            item_path = self.data[idx]
            audio_signal, audio_sr, audio_target, target_sr, config_dict = self.load_item_from_disk(item_path)

        signal1 = self._formatAndConvertToSpectogram(audio_signal, audio_sr)
        signal2 = self._delaySum(audio_signal, config_dict)
        #signal = torch.cat([signal1, signal2], dim=1)
        target = self._formatAndConvertToSpectogram(audio_target, target_sr)
        
        """istft = IStft(1, 1024, 256, "hann")
        flip_signal = np.einsum("ijk->kij", signal1.cpu().detach().numpy())
        new_signal = np.zeros((1, 256, flip_signal.shape[0]),dtype=np.float32)
        for index, item in enumerate(flip_signal):
            new_signal[...,index] = istft(item)"""
  
        #new_signal = torch.istft(signal1, n_fft=1024, hop_length= 256).float()
  
        test = self.waveform(signal2)
        torchaudio.save('./test_audio.wav', test.float() , 16000)
        return signal1, target
    
    def _delaySum(self, signal, config):
        freq_signal = self.transformation(signal)
        mic0 = list(np.subtract(config['microphones'][0], config['user_pos']))
        mic1 = list(np.subtract(config['microphones'][1], config['user_pos']))
        
        mic_array =  generate_mic_array({
            "mics": {
                "0": mic0,
                "1": mic1
            },
            "nb_of_channels": 2
        })

        X = freq_signal.cpu().detach().numpy()
        X = np.einsum("ijk->kij", X)
        delay = np.squeeze(get_delays_based_on_mic_array(np.array([config['source_dir']]), mic_array, 1024)) #set variable 1024
        
        signal = np.zeros((1, X.shape[2], X.shape[0]),dtype=complex)
        for index, item in enumerate(X):
            signal[...,index] = self.delay_and_sum(freq_signal=item, delay=delay)

        signal = torch.from_numpy(signal).to(self.device)
        #signal = torch.log(signal)
        return signal
        
    def _formatAndConvertToSpectogram(self, raw_signal, sr):
        signal = self._resample(raw_signal, sr)
        signal = self._cut(signal)
        signal = self._right_pad(signal)

        # get the log mean spectogram of the array of mics
        signal = self._set_mono(signal)
        freq_signal = self.transformation(raw_signal)
        #freq_signal = 20*torch.log10(torch.abs(freq_signal) + 1e-09)
        #freq_signal = torchaudio.functional.amplitude_to_DB(freq_signal, multiplier=20.0, amin=1e-10, db_multiplier=1.0 )

        return freq_signal

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
            signal = signal[:, :self.num_samples]
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
        audio_target_path = os.path.join(subfolder_path, 'target.wav')
        configs_file_path = os.path.join(subfolder_path, 'configs.yaml')

        # Get config dict
        with open(configs_file_path, "r") as stream:
            try:
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # Get audio file
        #audio_signal, fs = sf.read(audio_file_path)
        audio_signal, audio_sr = torchaudio.load(audio_file_path)
        audio_target, target_sr = torchaudio.load(audio_target_path)

        return audio_signal, audio_sr, audio_target, target_sr, config_dict

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