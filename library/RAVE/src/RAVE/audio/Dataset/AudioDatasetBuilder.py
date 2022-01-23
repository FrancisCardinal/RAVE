import glob
import os
import rir_generator as rir
import yaml
import numpy as np
from scipy import signal
import soundfile as sf

from pyodas.utils import TYPES
from pyodas.core import SpatialCov, Stft

SIDE_ID = 0         # X
DEPTH_ID = 1         # Y
HEIGHT_ID = 2         # Z

CONFIG_PATH = 'C:\\GitProjet\\RAVE\\library\\RAVE\\src\\RAVE\\audio\\Dataset\\DatasetBuilder_config.yaml'

SAMPLE_RATE = 16000
FRAME_SIZE = 512
STFT_WINDOW = 'hann'

SOUND_MARGIN = 0.5       # Assume every sound source is margins away from receiver and each other

class AudioDatasetBuilder:
    """
    Class which handles the generation of the audio dataset through the randomization of sources and various
    parameters passed through DatasetBuilder_config and input_path yaml files.

    Args:
        dataset_size (int): Number of dataset items to generate
    """

    user_pos = []
    receiver_height = 1.5
    receiver_rel = np.array(              # Receiver (microphone) positions relative to "user" [x, y, z] (m)
                    ([-0.05, 0, 0],
                     [0.05, 0, 0]))
    c = 340                                 # Sound velocity (m/s)
    reverb_time = 0.4                       # Reverberation time (s)
    nsample = 4096                          # Number of output samples

    def __init__(self, sources_path, noises_path, output_path, max_noise_sources, speech_noise):
        self.receiver_abs = None                        # Variable that will be used for receiver positions in room.
        self.max_noise_sources = max_noise_sources      # Maximum number of noise sources in one output segment
        self.speech_noise = speech_noise                # Bool controlling if noise sources can be speech
        self.max_source_distance = 5                    # Maximum distance from source to receiver

        # Stft class
        self.n_channels = len(self.receiver_rel)
        self.stft = Stft(self.n_channels, FRAME_SIZE, STFT_WINDOW)

        # Load params/configs
        with open(CONFIG_PATH, "r") as stream:
            try:
                self.configs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.rooms = self.configs['room']

        # Load input sources paths (speech, noise)
        self.source_paths = glob.glob(os.path.join(sources_path, '**', '*.wav'))
        self.noise_paths = glob.glob(os.path.join(noises_path, '**', '*.wav'))

        # Prepare output subfolder
        self.output_subfolder = output_path
        # if os.path.exists(self.output_subfolder):
        #     # TODO: Add functionality if existing output folder
        #     print(f"ERROR: Output folder '{self.output_subfolder}' already exists, exiting.")
        #     exit()

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
    def combine_sources(audios):
        """
        Method used to combine audio source with noises.

        Args:
            audios (list[list[ndarray]]): All audio sources. Shape is (audio_count, channels, [signal])
        Returns:
            combined_audio: Combined source with noises. Shape is like source (channels, [signal])
        """
        # Get length of shortest audio
        array_lengths = []
        for audio_array in audios:
            array_lengths.append(len(audio_array[0]))
        max_length = min(array_lengths)

        combined_audio = audios[0]
        for audio in audios[1:]:
            for combined_channel, noise_channel in zip(combined_audio, audio):
                combined_channel += noise_channel[:max_length]

        return combined_audio

    def generate_ground_truth(self, signal_x):
        """
        Determines the spectrogram of the input temporal signal through the  Short Term Fourier Transform (STFT)
        use as ground truth for dataset.

        Args:
            signal_x (ndarray): Signal on which to get spectrogram.
        Returns:
            stft_x: STFT of input signal x.
        """
        stft_x = self.stft(signal_x)
        return stft_x

    def generate_abs_receivers(self, room):
        """
        Generate absolute position for receivers.
        Only generates the receiver directly in the middle of the room for now.

        Args:
            room (list[float, float, float]): Room dimensions in which to place receivers (x, y, z) (m).
        """
        # Get only middle of x and y values, assume z is fixed at human height
        room_np = np.array([room[:-1]])
        self.user_pos = np.append(room_np / 2, self.receiver_height)

        # For every receiver, set x and y by room dimension and add human height as z
        self.receiver_abs = []
        for receiver in self.receiver_rel:
            receiver_center = receiver + self.user_pos
            self.receiver_abs.append(receiver_center.tolist())

    def generate_random_position(self, room, source_pos=None, multiple_noise=False):
        """
        Generates position for sound source (either main source or noise) inside room.
        Checks to not superpose source with receiver, and noise with either source and receiver.

        Args:
            room (ndarray): Dimension of room (max position).
            source_pos (ndarray): Position of source, in order to not superpose with noise (None if source).
            multiple_noise (bool): Generate multiple noise positions or not.
        Returns:
            Returns random position (or position list if more than one) for sound source.
        """
        if not source_pos:
            random_pos = np.array([np.random.rand(), np.random.rand(), self.receiver_height])

            # Add sources only in front of receiver as use-case (depth)
            random_pos[DEPTH_ID] *= room[DEPTH_ID] - self.user_pos[DEPTH_ID]
            random_pos[DEPTH_ID] += self.user_pos[DEPTH_ID] + SOUND_MARGIN
            # x
            random_pos[SIDE_ID] *= room[SIDE_ID]

            return random_pos

        else:
            # TODO: Check if more intelligent way to do than loop
            # TODO: Make sure noises are not superposed
            # Sources can be anywhere in room except on source or receiver
            random_pos_list = []
            noise_count = self.max_noise_sources if multiple_noise else 1
            for noise_i in range(noise_count):
                while len(random_pos_list) == noise_i:
                    random_pos = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
                    random_pos *= room

                    # If noise on source or on receiver, reroll
                    if self.receiver_height-SOUND_MARGIN <= random_pos[HEIGHT_ID] <= self.receiver_height+SOUND_MARGIN:

                        # Check if on receiver
                        side_user_bounds = [self.user_pos[SIDE_ID]-SOUND_MARGIN, self.user_pos[SIDE_ID]+SOUND_MARGIN]
                        depth_user_bounds = [self.user_pos[DEPTH_ID]-SOUND_MARGIN, self.user_pos[DEPTH_ID]+SOUND_MARGIN]
                        if side_user_bounds[0] <= random_pos[SIDE_ID] <= side_user_bounds[1] and \
                           depth_user_bounds[0] <= random_pos[DEPTH_ID] <= depth_user_bounds[1]:
                            continue

                        # Check if on source
                        side_src_bounds = [source_pos[SIDE_ID]-SOUND_MARGIN, source_pos[SIDE_ID]+SOUND_MARGIN]
                        depth_src_bounds = [source_pos[DEPTH_ID]-SOUND_MARGIN, source_pos[DEPTH_ID]+SOUND_MARGIN]
                        if side_src_bounds[0] <= random_pos[SIDE_ID] <= side_src_bounds[1] and \
                             depth_src_bounds[0] <= random_pos[DEPTH_ID] <= depth_src_bounds[1]:
                            continue

                    # If not on source or user, add position to random position list
                    random_pos_list.append(random_pos)

            return random_pos_list

    def get_random_noise(self, number_noises=None):
        if not number_noises:
            number_noises = self.max_noise_sources
        random_indices = np.random.randint(0, len(self.noise_paths), number_noises)
        noise_path_list = [self.noise_paths[i] for i in random_indices]
        return noise_path_list

    def save_files(self, combined_signal, combined_gt, source_name, source_gt,
                   noise_names, noise_gts, combined_noise_gt):
        """
        Save various files needed for dataset (see params).

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

    def generate_and_apply_rirs(self, source_audio, source_pos, room):
        """
        Function that generates Room Impulse Responses (RIRs) to source signal positions and applies them to signals.
        See <https://pypi.org/project/rir-generator/>. Will have to look later on at
        <https://github.com/DavidDiazGuerra/gpuRIR#simulatetrajectory> for moving sources.

        Args:
            source_audio (ndarray): Audio file.
            source_pos (list[float, float, float]): Source position ([x y z] (m))
            room (list[float, float, float]): Room dimensions ([x y z] (m))
        Returns:
            source_with_rir: Source signal with RIRs applied (shape is [channels,
        """
        # Generate RIRs
        # receivers = self.receiver_abs.to_list()
        rirs = rir.generate(
            c=self.c,                               # Sound velocity (m/s)
            fs=SAMPLE_RATE,                         # Sample frequency (samples/s)
            r=self.receiver_abs,                            # Receiver position(s) [x y z] (m)
            s=source_pos,                           # Source position [x y z] (m)
            L=room,                                 # Room dimensions [x y z] (m)
            reverberation_time=self.reverb_time,    # Reverberation time (s)
            nsample=self.nsample,                   # Number of output samples
        )

        # Apply RIR to signal
        source_with_rir = self.apply_rir(rirs, source_audio)

        return source_with_rir
