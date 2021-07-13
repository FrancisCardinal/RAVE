import os
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torch
from torch.nn.functional import pad


class UrbanSoundDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,  # Keep transformation abstract, in our instance we use mel
                 target_sample_rate,
                 num_samples,  # The amount of samples we want
                 device,
                 ):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    # required when deriving from Dataset class
    def __len__(self):  # Define how to calculate the length
        return len(self.annotations)

    def __getitem__(self, index):  # Define how to get an item
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        # Signal -> Tensor -> (num_channels, num_samples or data)
        signal, sample_rate = torchaudio.load(audio_sample_path)  # signal is the waveform, sample_rates are variable
        signal = signal.to(self.device)  # Register the signal on the device and use GPU if avail
        signal = self._resample_if_necessary(signal,
                                             sample_rate)  # We want all our signals to have the same sample rate to avoid non-uniform transformations
        signal = self._mix_down_if_necessary(signal)  # If we have a stereo signal we can a mono signal
        signal = self._right_pad_if_necessary(signal)  # If our signal isnt lenthy enough , we want to pad 0's
        signal = self._cut_if_necessary(signal)  # If our signal is too long, trim it
        signal = self.transformation(signal)  # TorchAudio transformations are functors so we can call them
        return signal, label

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
             signal = signal[:, :self.num_samples]  # Leave first dimension alone, 2nd dim keep until self.num_samples
        return signal

    def _right_pad_if_necessary(self, signal):
        signal_len = signal.shape[1]
        if signal_len < self.num_samples:
            num_missing_samples = self.num_samples - signal_len
            last_dim_padding = (0, num_missing_samples)  # Pad accepts a number of overloads we use
            # (prepend amount, append amount)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sample_rate):
        if sample_rate == self.target_sample_rate:
            return signal
        resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
        return resampler(signal)

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)  # A traditional way of dropping a stereo to a mono
        return signal

    def _get_audio_sample_path(self, index):
        fold_folder = f"fold{self.annotations.iloc[index, 5]}"
        file_name = self.annotations.iloc[index, 0]
        path = os.path.join(self.audio_dir, fold_folder, file_name)
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]


if __name__ == "__main__":
    ANNOTATIONS_FILE = "C:/Users/Jay/Desktop/TorchAudio/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "C:/Users/Jay/Desktop/TorchAudio/UrbanSound8K/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050  # If sample_rate = num_samples it means we want 1 sec of audio

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,  # Frame size
        hop_length=512,  # Frame size / 2
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
    print(f"There are {len(usd)} samples")
    signal, label = usd[0]
    a = 1
