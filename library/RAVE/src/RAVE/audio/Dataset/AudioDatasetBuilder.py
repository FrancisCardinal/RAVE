import glob
import os
import yaml

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import soundfile as sf
import rir_generator as rir
from pyodas.utils import sqrt_hann


SIDE_ID = 0         # X
DEPTH_ID = 1         # Y
HEIGHT_ID = 2         # Z

SAMPLE_RATE = 16000
FRAME_SIZE = 512
STFT_WINDOW = 'hann'

SOUND_MARGIN = 0.5       # Assume every sound source is margins away from receiver and each other


class AudioDatasetBuilder:
    """
    Class which handles the generation of the audio dataset through the randomization of sources and various
    parameters passed through DatasetBuilder_config and input_path yaml files.

    Args:
        sources_path (str): Path to sources directory.
        noises_path (str): Path to noise directory.
        output_path (str): Path to output directory.
        noise_count_range (list(int, int)): Range of number of noises.
        speech_noise (bool): Whether to use speech as noise.
        debug (bool): Run in debug mode.
    """

    user_pos = []
    source_direction = []
    room = []
    receiver_height = 1.5
    receiver_rel = np.array(                # Receiver (microphone) positions relative to "user" [x, y, z] (m)
                            ([-0.05, 0, 0],
                             [0.05, 0, 0])
                            )
    c = 340                                 # Sound velocity (m/s)
    reverb_time = 0.4                       # Reverberation time (s)
    n_sample = 4096                          # Number of output samples

    def __init__(self, sources_path, noises_path, output_path, noise_count_range, speech_noise, debug):
        self.receiver_abs = None
        self.noise_count_range = noise_count_range
        self.noise_count = noise_count_range[0]
        self.speech_noise = speech_noise
        self.max_source_distance = 5
        self.is_debug = debug

        self.n_channels = len(self.receiver_rel)

        # Load params/configs
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DatasetBuilder_config.yaml')
        with open(config_path, "r") as stream:
            try:
                self.configs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.rooms = self.configs['room']

        # Load input sources paths (speech, noise)
        self.source_paths = glob.glob(os.path.join(sources_path, '*.wav'))
        self.noise_paths = glob.glob(os.path.join(noises_path, '*.wav'))
        self.noise_speech_paths = self.noise_paths.copy()
        self.noise_speech_paths.extend(self.source_paths)

        # Prepare output subfolder
        self.output_subfolder = output_path
        os.makedirs(self.output_subfolder, exist_ok=True)

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
        audio_signal, fs = sf.read(file_path)

        # TODO: Find how to handle if sample rate not at 16 000 (current dataset is all ok)
        if fs != SAMPLE_RATE:
            print(f"ERROR: Sample rate of files ({fs}) do not concord with SAMPLE RATE={SAMPLE_RATE}")
            exit()

        return audio_signal

    @staticmethod
    def apply_rir(rirs, in_signal):
        """
        Method to apply RIRs to mono signals.

        Args:
            rirs (ndarray): RIRs generated previously to reflect room, source position and mic positions
            in_signal (ndarray): Monochannel input signal to be reflected to multiple microphones
        Returns:
            output(ndarray): Every RIR applied to monosignal
        """
        channels = rirs.shape[1]
        frames = in_signal[0].shape[0]
        output = np.empty((channels, frames))

        for channel_index in range(channels):
            output[channel_index] = signal.convolve(in_signal[0], rirs[:, channel_index])[:frames]
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
        max_length, small_array_idx = min(array_lengths), np.argmin(array_lengths)

        combined_audio = audios[small_array_idx]
        for audio in (audios[:small_array_idx] + audios[small_array_idx:]):
            for combined_channel, noise_channel in zip(combined_audio, audio):
                combined_channel += noise_channel[:max_length]

        return combined_audio

    def generate_ground_truth(self, signal_x, signal_name=''):
        """
        Determines the spectrogram of the input temporal signal through the  Short Term Fourier Transform (STFT)
        use as ground truth for dataset.

        Args:
            signal_x (ndarray): Signal on which to get spectrogram.
            signal_name (str): Name of signal to plot on debug spectrogram.
        Returns:
            stft_x: STFT of input signal x.
        """
        frame_size = 1024
        chunk_size = frame_size // 2
        window = sqrt_hann(chunk_size)

        f, t, stft_x = signal.stft(signal_x[0], SAMPLE_RATE, window, chunk_size, nfft=frame_size)

        if self.is_debug:
            plt.pcolormesh(t, f, np.abs(stft_x), shading='gouraud', title=signal_name)
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show()

        return stft_x

    def generate_abs_receivers(self, room):
        """
        Generate absolute position for receivers.
        Only generates the receiver directly in the middle of the room for now.

        Args:
            room (list[float, float, float]): Room dimensions in which to place receivers (x, y, z) (m).
        """
        # Get only middle of x and y values, assume z is fixed at human height
        self.room = room
        room_np = np.array([room[:-1]])
        self.user_pos = np.append(room_np / 2, self.receiver_height)

        # For every receiver, set x and y by room dimension and add human height as z
        self.receiver_abs = []
        for receiver in self.receiver_rel:
            receiver_center = receiver + self.user_pos
            self.receiver_abs.append(receiver_center.tolist())

    def generate_random_position(self, room, source_pos=np.array([])):
        """
        Generates position for sound source (either main source or noise) inside room.
        Checks to not superpose source with receiver, and noise with either source and receiver.

        Args:
            room (ndarray): Dimension of room (max position).
            source_pos (ndarray): Position of source, in order to not superpose with noise (None if source).
        Returns:
            Returns random position (or position list if more than one) for sound source.
        """
        if source_pos.size == 0:
            random_pos = np.array([np.random.rand(), np.random.rand(), self.receiver_height])

            # Add sources only in front of receiver as use-case (depth)
            random_pos[DEPTH_ID] *= (room[DEPTH_ID] - self.user_pos[DEPTH_ID] - SOUND_MARGIN)
            random_pos[DEPTH_ID] += self.user_pos[DEPTH_ID] + SOUND_MARGIN
            # x
            random_pos[SIDE_ID] *= room[SIDE_ID]

            self.source_direction = random_pos - self.user_pos

            return random_pos

        else:
            # TODO: Make sure noises are not superposed (maybe useful?)
            # Sources can be anywhere in room except on source or receiver
            random_pos_list = []
            for noise_i in range(self.noise_count):
                # TODO: Check if more intelligent way to do than loop
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
        """
        Gets random noises to be added to audio clip.

        Args:
            number_noises: Number of noises to use (if specified, overrides self.noise_count).

        Returns:
            List of paths to noises (and/or speech) to use.
        """
        # Set noise count for this round
        if number_noises:
            self.noise_count = number_noises
        else:
            self.noise_count -= self.noise_count[0] + 1
            self.noise_count = self.noise_count % (self.noise_count[1]-self.noise_count[0]) + self.noise_count[0]

        # Use noise + speech list if allowed in args
        if self.speech_noise:
            noise_list = self.noise_speech_paths
        else:
            noise_list = self.noise_paths

        # Get random indices and return items in new list
        random_indices = np.random.randint(0, len(noise_list), self.noise_count)
        noise_path_list = [noise_list[i] for i in random_indices]
        return noise_path_list

    def save_files(self, combined_signal, combined_gt, source_name, source_gt, source_pos,
                   noise_names, noise_pos, noise_gts, combined_noise_gt):
        """
        Save various files needed for dataset (see params).

        Args:
            combined_signal: Audio signal array of sources and noise together, 1 channel per receiver.
            combined_gt: STFT of combined audio signal.
            source_name: Name of source sample used.
            source_gt: STFT of source signal.
            source_pos: Source position.
            noise_names: List of names of noise samples.
            noise_pos: List of noise positions.
            noise_gts: List of STFT of noise signals.
            combined_noise_gt: STFT of combined noise signals.
        Returns:
            subfolder_path (str): String containing path to newly created dataset subfolder.
        """
        # Create subfolder
        subfolder_path = os.path.join(self.output_subfolder, f'{len(noise_names)}', source_name)
        for noise_name in noise_names:
            subfolder_path += '_' + noise_name
        subfolder_index = 1
        if os.path.exists(subfolder_path):
            while os.path.exists(subfolder_path + f'_{subfolder_index}'):
                subfolder_index += 1
            subfolder_path += f'_{subfolder_index}'
        os.makedirs(subfolder_path, exist_ok=True)

        # Save audio file and audio ground truth
        audio_file_name = os.path.join(subfolder_path, 'audio.wav')
        sf.write(audio_file_name, combined_signal.T, SAMPLE_RATE)
        audio_gt_name = os.path.join(subfolder_path, 'combined_audio_gt.npz')
        np.savez_compressed(audio_gt_name, combined_gt)

        # Save source ground truth
        source_gt_name = os.path.join(subfolder_path, 'source.npz')
        np.savez_compressed(source_gt_name, source_gt)

        # Save combined noise ground truth
        combined_noise_gt_name = os.path.join(subfolder_path, 'noise.npz')
        np.savez_compressed(combined_noise_gt_name, combined_noise_gt)

        # Save noise ground truth in debug
        if self.is_debug:
            for noise_name, noise_gt in zip(noise_names, noise_gts):
                noise_gt_name = os.path.join(subfolder_path, f'{noise_name}.npz')
                np.savez_compressed(noise_gt_name, noise_gt)

        # Save yaml file with configs
        config_dict_file_name = os.path.join(subfolder_path, 'configs.yaml')
        config_dict = dict(
            path=subfolder_path,
            n_channels=self.n_channels,
            microphones=self.receiver_abs,
            room=self.room,
            user_pos=self.user_pos.tolist(),
            source_pos=np.around(source_pos, 3).tolist(),
            noise_pos=[np.around(i, 3).tolist() for i in noise_pos],
            source=source_name,
            source_dir=np.around(self.source_direction, 3).tolist(),
            noise=noise_names
        )
        with open(config_dict_file_name, 'w') as outfile:
            yaml.dump(config_dict, outfile, default_flow_style=None)

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
            r=self.receiver_abs,                    # Receiver position(s) [x y z] (m)
            s=source_pos,                           # Source position [x y z] (m)
            L=room,                                 # Room dimensions [x y z] (m)
            reverberation_time=self.reverb_time,    # Reverberation time (s)
            nsample=self.n_sample,                   # Number of output samples
        )

        # Apply RIR to signal
        source_with_rir = self.apply_rir(rirs, source_audio)

        return source_with_rir


    def generate_single_file(self):
        # TODO: GENERATE A SINGLE FILE FOR ON THE RUN GENERATION
        pass