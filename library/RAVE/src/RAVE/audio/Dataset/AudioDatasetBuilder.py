import glob
import os
import rir_generator as rir
import yaml
import numpy as np
from scipy import signal
import soundfile as sf

from pyodas.utils import TYPES
from pyodas.core import SpatialCov

SAMPLE_RATE = 16000


class AudioDatasetBuilder:
    """
    Class which handles the generation of the audio dataset through the randomization of sources and various
    parameters passed through DatasetBuilder_config and input_path yaml files.

    Args:
        dataset_size (int): Number of dataset items to generate

    """

    r = [               # Receiver (microphone) positions [x, y, z] (m)

    ]
    c = 340             # Sound velocity (m/s)
    reverb_time = 0.4   # Reverberation time (s)
    nsample = 4096      # Number of output samples

    def __init__(self, max_sources, sources_path, output_path, noises_path):
        self.max_sources = max_sources      # Maximum number of sources in one output segment

        # Load params/configs
        with open("DatasetBuilder_config.yaml", "r") as stream:
            try:
                self.configs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.positions = self.configs['position']
        self.rooms = self.configs['room']

        # Load input sources paths (speech, noise)
        self.source_paths = glob.glob(os.path.join(sources_path, '**', '*.wav'))
        self.noise_paths = glob.glob(os.path.join(noises_path, '**', '*.wav'))

        # Prepare output subfolder
        self.output_subfolder = output_path
        if os.path.exists(self.output_subfolder):
            # TODO: Add functionality if existing output folder
            print(f"ERROR: Output folder '{self.output_subfolder}' already exists, exiting.")
            exit()


    @staticmethod
    def read_audio_file(file_path):
        """
        Reads and extracts audio from a file. Uses PyODAS WavSource class
        <https://introlab.github.io/pyodas/_build/html/pyodas/io/io.sources.html#module-pyodas.io.sources.wav_source>.

        Args:
            file_path (str): Path to audio file

        Returns:
            audio_signal (ndarray): Audio read from path of length chunk_size
            sample_frequency (int): Audio file sample frequency
        """
        audio_signal = sf.read(file_path, samplerate=SAMPLE_RATE)
        return audio_signal

    @staticmethod
    def apply_rir(rirs, signal):
        """
        Method to apply RIRs to mono signals.

        Args:
            rirs (ndarray): RIRs generated previously to reflect room, source position and mic positions
            signal (ndarray): Monochannel signal to be reflected to multiple microphones

        Returns:
            output(ndarray): Every RIR applied to monosignal
        """
        channels = rirs.shape[0]
        frames = signal.shape[0]
        output = np.empty((channels, frames))

        for channel_index in range(channels):
            output[channel_index] = signal.convolve(signal, rir[channel_index])[:frames]
        return output

    @staticmethod
    def combine_sources(source, noises, channels=6):
        """
        Method used to combine audio source with noises.

        Args:
            source (list[ndarray]): Source to add noises to. Shape is (channels, [signal])
            noises (list[list[ndarray]]): List of noises to add to source. Shape is (# noises, channels, [signal])
            channels (int): Number of audio channels.

        Returns:
            combined_audio: Combined source with noises. Shape is like source (channels, [signal])
        """
        # Get length of shortest audio
        array_lengths = [len(source[0])]
        for noise_array in noises:
            array_lengths.append(len(noise_array))
        max_length = min(array_lengths)

        combined_audio = source
        for noise in noises:
            for combined_channel, noise_channel in zip(combined_audio, noise):
                combined_channel += noise_channel[:max_length]

        return combined_audio

    @staticmethod
    def generate_ground_truth(x):
        """
        Determines the spatial covariance matrix of input signal to use as ground truth for dataset.

        Args:
            signal (ndarray): Signal on which to get spatial covariance matrix.

        Returns:
            spatial_covariance: Spatial covariance of signal.
        """
        X = np.fft.rfft(x)
        channels, bins = X.shape
        # TODO: Change scm for stft
        scm = SpatialCov(channels, (bins-1)*2)
        spatial_covariance = scm(X)

        return spatial_covariance

    def save_files(self, combined_signal, combined_gt, source_name, source_gt,
                   noise_names, noise_gts, combined_noise_gt):
        """
        Save various files needed for dataset.

        Args:
            combined_signal: Audio signal array of sources and noise together, 1 channel per receiver.
            combined_gt: STFT of combined audio signal.
            source_name: Name of source sample used.
            source_gt: STFT of source signal.
            noise_names: List of names of noise samples.
            noise_gts: List of STFT of noise signals.
            combined_noise_gt: STFT of combined noise signals.

        Returns:
            subfolder_path (str): String containing path to newly created dataset subfolder.
        """
        # Create subfolder
        subfolder_path = os.path.join(self.output_subfolder, '1', source_name)
        for noise_name in noise_names:
            subfolder_path = os.path.join(subfolder_path, noise_name)
        os.makedirs(subfolder_path, exist_ok=True)

        # Save audio file and audio ground truth
        audio_file_name = os.path.join(subfolder_path, 'combined_audio.wav')
        sf.write(audio_file_name, combined_signal, SAMPLE_RATE)
        audio_gt_name = os.path.join(subfolder_path, 'combined_audio_scm.npz')
        np.savez_compressed(audio_gt_name, combined_gt)

        # Save source ground truth
        source_gt_name = os.path.join(subfolder_path, f'{source_name}_gt.npz')
        np.savez_compressed(source_gt_name, source_gt)

        # Save noise and combined noise ground truths
        combined_noise_gt_name = os.path.join(subfolder_path, f'combined_noise_gt.npz')
        np.savez_compressed(combined_noise_gt_name, combined_noise_gt)
        for noise_name, noise_gt in zip(noise_names, noise_gts):
            noise_gt_name = os.path.join(subfolder_path, f'{noise_name}_gt.npz')
            np.savez_compressed(noise_gt_name, noise_gt)

        return subfolder_path

    def generate_and_apply_rirs(self, source_path, source_pos, room):
        """
        Function that generates Room Impulse Responses (RIRs) to source signal positions and applies them to signals.
        See <https://pypi.org/project/rir-generator/>. Will have to look later on at
        <https://github.com/DavidDiazGuerra/gpuRIR#simulatetrajectory> for moving sources.

        Args:
            source_path (str): Path to audio file.
            source_pos (list[float, float, float]): Source position ([x y z] (m))
            room (list[float, float, float]): Room dimensions ([x y z] (m))

        Returns:
            source_with_rir: Source signal with RIRs applied (shape is [channels,

        """
        s_signal = self.read_audio_file(source_path)
        rirs = rir.generate(
            c=self.c,                               # Sound velocity (m/s)
            fs=SAMPLE_RATE,                         # Sample frequency (samples/s)
            r=self.r,                               # Receiver position(s) [x y z] (m)
            s=source_pos,                           # Source position [x y z] (m)
            L=room,                                 # Room dimensions [x y z] (m)
            reverberation_time=self.reverb_time,    # Reverberation time (s)
            nsample=self.nsample,                   # Number of output samples
        )
        source_with_rir = self.apply_rir(rirs, s_signal)
        return source_with_rir
