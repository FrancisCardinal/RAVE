import torch
import torchaudio
from pyodas.utils import generate_mic_array, get_delays_based_on_mic_array

from tkinter import filedialog
import glob
import os
import random
import yaml
import numpy as np
import math

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
            sample_rate=16000,
            num_samples=32000
    ):
        self.duration = num_samples / sample_rate
        self.is_debug = IS_DEBUG
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.hop_len = 256
        self.frame_size = 2*self.hop_len
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

        self.get_dataset(self.dataset_path)

        self.generate_dataset_runtime = generate_dataset_runtime

        # # Split dataset into subsets
        # if not data_split:
        #     data_split = self.configs['data_split']
        # random.shuffle()

        self.transformation = torchaudio.transforms.Spectrogram(
            n_fft=self.frame_size,
            hop_length=self.hop_len,
            power=None,
            return_complex=True,
            window_fn= self.sqrt_hann_window
        )
        self.waveform = torchaudio.transforms.InverseSpectrogram(
            n_fft=self.frame_size,
            hop_length=self.hop_len,
            window_fn= self.sqrt_hann_window
        )
        self.delay_and_sum = Beamformer('delaysum', frame_size=self.frame_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get signals
        audio_signal, audio_sr, noise_target, noise_sr, speech_target, speech_sr, config_dict = self.load_item_from_disk(
            self.data[idx])

        # Get random begin point in signals
        min_val = min(audio_signal.shape[1], noise_target.shape[1], speech_target.shape[1])
        begin = random.randint(0, (min_val - self.num_samples)) if min_val > self.num_samples + 1 else 0

        # Pre Process
        sm = self.transformation(self.reformat(speech_target, speech_sr, begin))
        bm = self.transformation(self.reformat(noise_target, noise_sr, begin))
        xm = sm + bm

        c = torch.sum(torch.abs(bm) ** 2, dim=0) / (torch.sum(torch.abs(sm) ** 2 + torch.abs(bm) ** 2, dim=0) + 1e-09) # target


        yBf = 10 * torch.log10(torch.abs(self._delaySum(xm, config_dict))**2 + 1e-09)
        yAvg = 10 * torch.log10(torch.unsqueeze(torch.sum(torch.abs(xm)**2, dim=0), dim=0) + 1e-09)
        input_dnn = torch.cat([yAvg, yBf], dim=1)
        return input_dnn, torch.squeeze(c), yAvg, self.reformat(speech_target, speech_sr, begin), xm
    @staticmethod
    def sqrt_hann_window(window_length, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False):
        return torch.sqrt(torch.hann_window(window_length, periodic=periodic, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad))

    def reformat(self, raw_signal, sr, begin):
        signal = self._resample(raw_signal, sr)
        signal = self._cut(signal, begin)
        signal = self._right_pad(signal)
        return signal

    def _delaySum(self, xm, config):
        mic = config['mic_rel']

        mic_array = generate_mic_array({
            "mics": {
                "0": mic[0],
                "1": mic[1],
                "2": mic[2],
                "3": mic[3],
                "4": mic[4],
                "5": mic[5],
                "6": mic[6],
                "7": mic[7]
            },
            "nb_of_channels": 8
        })

        X = xm.cpu().detach().numpy()
        X = np.einsum("ijk->kij", X)
        delay = np.squeeze(
            get_delays_based_on_mic_array(np.array([config['source_dir']]), mic_array, self.frame_size))  # set variable 1024

        signal = np.zeros((1, X.shape[2], X.shape[0]), dtype=complex)
        for index, item in enumerate(X):
            signal[..., index] = self.delay_and_sum(freq_signal=item, delay=delay)

        signal = torch.from_numpy(signal)
        return signal

    def _resample(self, signal, sr):
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            signal = resampler(signal)
        return signal

    def _set_mono(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut(self, signal, begin):
        # begin = 0
        if signal.shape[1] > self.num_samples:
            signal = signal[:, begin: begin + self.num_samples]
        return signal

    def _right_pad(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            signal = torch.nn.functional.pad(signal, [0, num_missing_samples])
        return signal

    def get_dataset(self, dataset_path):
        """
        Used to get all data from a directory to perform further computations
        Args:
            dataset_path (string): Path where .wav files to train are located

        """

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
        file_count, dataset_list = self.dataset_builder.generate_sim_dataset(save_run=True)
        print(f"DATASET: Dataset generated at '{dataset_path}'. {file_count} files generated")
        dataset_list = np.array([str(i) for i in dataset_list], dtype=np.str)
        self.data = dataset_list

    @staticmethod
    def load_item_from_disk(subfolder_path):
        """
        Used to load a sample with speech.wav, noise.wav, audio.wav and configs.yaml from disk
        Args:
            subfolder_path (string): path where the sample is located on disk

        Returns:
            tuple: audio signal, audio sample rate, noise signal, noise sample rate, speech signal, speech sample rate,
                signal configuration dictionary
        """
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
