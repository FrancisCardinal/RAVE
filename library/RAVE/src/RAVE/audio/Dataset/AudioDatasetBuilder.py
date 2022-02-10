import glob
import os
import yaml
from tqdm import tqdm

import audiolib

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pyroomacoustics as pra

import soundfile as sf
import rir_generator as rir
from pyodas.utils import sqrt_hann


SIDE_ID = 0         # X
DEPTH_ID = 1         # Y
HEIGHT_ID = 2         # Z

SAMPLE_RATE = 16000
FRAME_SIZE = 1024
SOUND_MARGIN = 0.5       # Assume every sound source is margins away from receiver and each other


class AudioDatasetBuilder:
    """
    Class which handles the generation of the audio dataset through the randomization of sources and various
    parameters passed through DatasetBuilder_config and input_path yaml files.

    Args:
        sources_path (str): Path to sources directory. If None, gets folder from config file.
        noises_path (str): Path to noise directory. If None, gets folder from config file.
        output_path (str): Path to output directory.
        noise_count_range (list(int, int)): Range of number of noises.
        speech_noise (bool): Whether to use speech as noise.
        debug (bool): Run in debug mode.
    """

    user_pos = []
    source_direction = []
    current_room = []
    receiver_height = 1.5
    receiver_rel = np.array((                   # Receiver (microphone) positions relative to "user" [x, y, z] (m)
                                [-0.05, 0, 0],
                                [0.05, 0, 0]
                            ))
    c = 340                                     # Sound velocity (m/s)
    reverb_time = 0.4                           # Reverberation time (s)
    n_sample = 4096                             # Number of output samples

    def __init__(self, sources_path, noises_path, output_path, noise_count_range,
                 speech_noise, sample_per_speech, debug):

        # Set object values from arguments
        self.dir_noise_count_range = [noise_count_range[0], noise_count_range[1] + 1]
        self.speech_noise = speech_noise
        self.sample_per_speech = sample_per_speech
        self.is_debug = debug

        self.receiver_abs = None
        self.dir_noise_count = noise_count_range[0]
        self.max_source_distance = 5
        self.n_channels = len(self.receiver_rel)

        # Load params/configs
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_config.yaml')
        with open(config_path, "r") as stream:
            try:
                self.configs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.rooms = self.configs['room']
        self.banned_noises = self.configs['banned_noises']
        self.diffuse_noises = self.configs['diffuse_noises']
        self.max_diffuse_noises = self.configs['max_diffuse_noises']
        self.snr_limits = self.configs['snr_limits']

        # Load input sources paths (speech, noise)
        self.source_paths = glob.glob(os.path.join(sources_path, '*.wav'))
        self.noise_paths = glob.glob(os.path.join(noises_path, '*.wav'))

        # Split noise paths (diffuse, directional) and remove banned noises
        self.dif_noise_paths = []
        self.dir_noise_paths = []
        for noise in self.noise_paths:
            if any(banned_noise in noise for banned_noise in self.banned_noises):
                continue
            if any(banned_noise in noise for banned_noise in self.diffuse_noises):
                self.dif_noise_paths.append(noise)
            else:
                self.dir_noise_paths.append(noise)

        # Add speech to directional if in arguments
        if self.speech_noise:
            self.dir_noise_paths.extend(self.source_paths)

        # Prepare output subfolder
        self.output_subfolder = output_path
        os.makedirs(self.output_subfolder, exist_ok=True)

        # Prepare lists for run-time generation
        self.dataset_list = []

    @staticmethod
    def read_audio_file(file_path):
        """
        Reads and extracts audio from a file.

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
        frames = len(in_signal)
        output = np.empty((channels, frames))

        for channel_index in range(channels):
            output[channel_index] = signal.convolve(in_signal, rirs[:, channel_index])[:frames]
        return output

    @staticmethod
    def truncate_sources(source, dir_noise_list, dif_noise_list):
        """
        Method used to truncate audio sources to smallest one.

        Args:
            source (list[ndarray]): Speech source. Shape is (channels, [signal])
            dir_noise_list (list[list[ndarray]]): Directional noise sources. Shape is (noise_count, channels, [signal])
            dif_noise_list (list[list[ndarray]]): Diffuse noise sources. Shape is (noise_count, channels, [signal])
        Returns:
            Truncated speech source and noise source list
        """
        # Get length of shortest audio
        shortest = len(source)
        for noise in dir_noise_list:
            if len(noise) < shortest:
                shortest = len(noise)
        for noise in dif_noise_list:
            if len(noise) < shortest:
                shortest = len(noise)

        trunc_source = source[:shortest]
        trunc_dir_noise_list = []
        trunc_dif_noise_list = []
        for noise in dir_noise_list:
            trunc_noise = noise[:shortest]
            trunc_dir_noise_list.append(trunc_noise)
        for noise in dif_noise_list:
            trunc_noise = noise[:shortest]
            trunc_dif_noise_list.append(trunc_noise)

        return trunc_source, trunc_dir_noise_list, trunc_dif_noise_list

    @staticmethod
    def generate_spectogram(signal_x, mono=False, title=''):
        """
        Determines the spectrogram of the input temporal signal through the  Short Term Fourier Transform (STFT)
        use as ground truth for dataset.

        Args:
            signal_x (ndarray): Signal on which to get spectrogram.
            title (str): Title used for spectrogram plot (in debug mode only).
        Returns:
            stft_list: List of channels of STFT of input signal x.
        """
        chunk_size = FRAME_SIZE // 2
        window = sqrt_hann(FRAME_SIZE)

        if mono:
            signal_x = [signal_x]

        # Generate plot
        fig, ax = plt.subplots(len(signal_x), constrained_layout=True)
        fig.suptitle(title)
        fig.supylabel('Frequency [Hz]')
        fig.supxlabel('Time [sec]')

        stft_list = []
        for channel_idx, channel in enumerate(signal_x):
            # Get stft for every channel
            f, t, stft_x = signal.spectrogram(channel, SAMPLE_RATE, window, FRAME_SIZE, chunk_size)
            stft_log = 10*np.log10(stft_x)
            f_log = np.logspace(0, 4, len(f))
            stft_list.append(stft_x)

            # Add log and non-log plots
            if mono:
                im = ax.pcolormesh(t, f_log, stft_log, shading='gouraud')
                ax.set_yscale('log')
                fig.colorbar(im, ax=ax)
            else:
                im = ax[channel_idx].pcolormesh(t, f_log, stft_log, shading='gouraud')
                ax[channel_idx].set_ylabel(f'Channel_{channel_idx}')
                ax[channel_idx].set_yscale('log')
                fig.colorbar(im, ax=ax[channel_idx])

        plt.show()

        return stft_list

    @staticmethod
    def combine_sources(audios, snr=None):
        """
        Method used to combine audio source with noises.

        Args:
            audios (list[list[ndarray]]): All audio sources. Shape is (audio_count, channels, [signal])
            snr (float): Signal to Noise Ratio. Also indicated audios given are clean + noise.
        Returns:
            combined_audio: Combined source with noises. Shape is like source (channels, [signal])
        """
        combined_audio = audios[0]
        source_audio = audios[0]
        noise_audio = audios[1]
        # If given an SNR, use audiolib function to add noise to clean. If not, just add noises together
        if snr:
            for c, (combined_channel, noise_channel, source_channel) in \
                    enumerate(zip(combined_audio, noise_audio, source_audio)):
                source_audio[c], noise_audio[c], combined_audio[c] = \
                    audiolib.snr_mixer(source_channel, noise_channel, snr)
        else:
            for audio in (audios[1:]):
                audio_copy = audio.copy()
                for combined_channel, noise_channel in zip(combined_audio, audio_copy):
                        combined_channel += noise_channel

        return combined_audio, noise_audio, source_audio

    def plot_3d_room(self, source_pos, source_name, noise_pos_list, noise_name_list):
        """
        Visualize the virtual room by plotting.

        Args:
            source_pos (ndarray): Position of speech source.
            source_name (str): Name of speech source.
            noise_pos_list (list[ndarray]): List of positions of noise sources.
            noise_name_list (list[str]):  List of names of noise sources.

        """
        # TODO: VARY COLOR INTENSITY BY SNR
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlabel('Side (x)')
        ax.set_ylabel('Depth (y)')
        ax.set_zlabel('Height (z)')

        # # Room
        # r = [
        #     [0, self.current_room[SIDE_ID]],
        #     [0, self.current_room[DEPTH_ID]],
        #     [0, self.current_room[HEIGHT_ID]]
        # ]
        # for idx, (s, e) in enumerate(combinations(np.array(list(product(r[0], r[1], r[2]))), 2)):
        #     if np.sum(np.abs(s - e)) == r[idx][1] - r[idx][0]:
        #         ax.plot3D(*zip(s, e), color="gray", linestyle='--')
        #
        # # User and microphones
        # u = [
        #     [self.user_pos[SIDE_ID] - SOUND_MARGIN, self.user_pos[SIDE_ID] + SOUND_MARGIN],
        #     [self.user_pos[DEPTH_ID] - SOUND_MARGIN, self.user_pos[DEPTH_ID] + SOUND_MARGIN],
        #     [self.user_pos[HEIGHT_ID] - SOUND_MARGIN, self.user_pos[HEIGHT_ID] + SOUND_MARGIN]
        # ]
        # for idx, (s, e) in enumerate(combinations(np.array(list(product(r[0], u[1], u[2]))), 2)):
        #     if np.sum(np.abs(s - e)) == u[idx][1] - u[idx][0]:
        #         ax.plot3D(*zip(s, e), color="b", linestyle=':')
        for mic_pos in self.receiver_abs:
            ax.scatter3D(mic_pos[SIDE_ID], mic_pos[DEPTH_ID], mic_pos[HEIGHT_ID], cmap='Blues')
        ax.text(mic_pos[SIDE_ID], mic_pos[DEPTH_ID], mic_pos[HEIGHT_ID], 'User')

        # Source
        ax.scatter3D(source_pos[SIDE_ID], source_pos[DEPTH_ID], source_pos[HEIGHT_ID], cmap='Greens')
        ax.text(source_pos[SIDE_ID], source_pos[DEPTH_ID], source_pos[HEIGHT_ID], source_name)
        # ADD SNR WHERE c = SNR
        # ax.scatter3D(source_pos[SIDE_ID], source_pos[DEPTH_ID], source_pos[HEIGHT_ID], c=SNR, cmap='Greens')

        # Noise
        for noise_pos, noise_name in zip(noise_pos_list, noise_name_list):
            ax.scatter3D(noise_pos[SIDE_ID], noise_pos[DEPTH_ID], noise_pos[HEIGHT_ID], cmap='Reds')
            # ax.scatter3D(noise_pos[SIDE_ID], noise_pos[DEPTH_ID], noise_pos[HEIGHT_ID], c=SNR, cmap='Reds')
            ax.text(noise_pos[SIDE_ID], noise_pos[DEPTH_ID], noise_pos[HEIGHT_ID], noise_name)

        plt.show()

    def set_user_position(self, room):
        """
        Generate absolute position for user (and for receivers).

        Args:
            room (list[float, float, float]): Room dimensions in which to place receivers (x, y, z) (m).
        """
        self.current_room = room

        # Get random position and assign x and y wth sound margins (user not stuck on wall)
        random_pos = np.array([np.random.rand(), np.random.rand(), self.receiver_height])
        random_pos[SIDE_ID] = random_pos[SIDE_ID] * (room[SIDE_ID] - SOUND_MARGIN*2) + SOUND_MARGIN
        random_pos[DEPTH_ID] = random_pos[DEPTH_ID] * (room[DEPTH_ID] - SOUND_MARGIN*2) + SOUND_MARGIN
        self.user_pos = random_pos

        # For every receiver, set x and y by room dimension and add human height as z
        self.receiver_abs = []
        for receiver in self.receiver_rel:
            receiver_center = receiver + self.user_pos
            self.receiver_abs.append(receiver_center.tolist())

    def get_random_position(self, room, source_pos=np.array([])):
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
            for noise_i in range(self.dir_noise_count):
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

    def get_random_noise(self, number_noises=None, diffuse_noise=False):
        """
        Gets random noises to be added to audio clip.

        Args:
            number_noises (int): Number of noises to use (if specified, overrides self.noise_count).
            diffuse_noise (bool): Bool to use diffuse or directional noise.

        Returns:
            List of paths to noises (and/or speech) to use.
        """

        if diffuse_noise:
            random_indices = np.random.randint(0, len(self.dif_noise_paths), self.max_diffuse_noises)
            noise_path_list = [self.dif_noise_paths[i] for i in random_indices]
        else:
            # Set noise count for this round
            if number_noises:
                self.dir_noise_count = number_noises
            else:
                self.dir_noise_count -= self.dir_noise_count_range[0]
                self.dir_noise_count += 1
                self.dir_noise_count = self.dir_noise_count % (self.dir_noise_count_range[1] - self.dir_noise_count_range[0])
                self.dir_noise_count += self.dir_noise_count_range[0]

            # Get random indices and return items in new list
            random_indices = np.random.randint(0, len(self.dir_noise_paths), self.dir_noise_count)
            noise_path_list = [self.dir_noise_paths[i] for i in random_indices]

        return noise_path_list

    def save_files(self, combined_signal, source_signal, combined_noise, target_signal,
                   source_name, source_pos, noise_names, noise_pos, save_spec=False, snr=1):
        """
        Save various files needed for dataset (see params).

        Args:
            combined_signal: Audio signal array of sources and noise together, 1 channel per receiver.
            source_signal: Audio signal array of source, 1 channel per receiver.
            combined_noise: Audio signal array of noises together, 1 channel per receiver.
            target_signal: Audio signal array of target, single channel.
            source_name: Name of source sample used.
            source_pos: Source position.
            noise_names: List of names of noise samples.
            noise_pos: List of noise positions.
            save_spec (bool): Control if saving spectrograms or not.
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

        # Save combined audio
        audio_file_name = os.path.join(subfolder_path, 'audio.wav')
        sf.write(audio_file_name, combined_signal.T, SAMPLE_RATE)
        if self.is_debug:
            combined_gt = self.generate_spectogram(combined_signal, title='Audio')
            if save_spec:
                audio_gt_name = os.path.join(subfolder_path, 'audio.npz')
                np.savez_compressed(audio_gt_name, combined_gt)

        # Save target
        target_file_name = os.path.join(subfolder_path, 'target.wav')
        sf.write(target_file_name, target_signal, SAMPLE_RATE)
        if self.is_debug:
            # Save source (speech)
            target_gt = self.generate_spectogram(target_signal, mono=True, title='Target')
            if save_spec:
                target_gt_name = os.path.join(subfolder_path, 'target.npz')
                np.savez_compressed(target_gt_name, target_gt)

        # Save source (with rir)
        speech_file_name = os.path.join(subfolder_path, 'speech.wav')
        sf.write(speech_file_name, source_signal.T, SAMPLE_RATE)
        if self.is_debug:
            source_gt = self.generate_spectogram(source_signal, title='Speech')
            if save_spec:
                source_gt_name = os.path.join(subfolder_path, 'speech.npz')
                np.savez_compressed(source_gt_name, source_gt)

        # Save combined noise
        noise_file_name = os.path.join(subfolder_path, 'noise.wav')
        sf.write(noise_file_name, combined_noise.T, SAMPLE_RATE)
        if self.is_debug:
            noise_gt = self.generate_spectogram(combined_noise, title='Noise')
            if save_spec:
                noise_gt_name = os.path.join(subfolder_path, 'noise.npz')
                np.savez_compressed(noise_gt_name, noise_gt)

        # Save yaml file with configs
        config_dict_file_name = os.path.join(subfolder_path, 'configs.yaml')
        config_dict = self.generate_config_dict(subfolder_path, source_pos, noise_pos, source_name, noise_names, snr)
        with open(config_dict_file_name, 'w') as outfile:
            yaml.dump(config_dict, outfile, default_flow_style=None)

        return subfolder_path

    def generate_and_apply_rirs(self, source_audio, source_pos, room, diffuse=False):
        """
        Function that generates Room Impulse Responses (RIRs) to source signal positions and applies them to signals.
        See <https://pypi.org/project/rir-generator/>.
        Will have to look later on at
        <https://github.com/DavidDiazGuerra/gpuRIR#simulatetrajectory> for moving sources.

        Args:
            source_audio (ndarray): Audio file.
            source_pos (ndarray): Source position ([x y z] (m))
            room (list[float, float, float]): Room dimensions ([x y z] (m))
            diffuse (bool): Whether input source is diffuse noise or not.
        Returns:
            source_with_rir: Source signal with RIRs applied (shape is [channels,
        """
        # Generate RIRs
        # TODO: MOVING RIR
        rirs = rir.generate(
            c=self.c,                               # Sound velocity (m/s)
            fs=SAMPLE_RATE,                         # Sample frequency (samples/s)
            r=self.receiver_abs,                    # Receiver position(s) [x y z] (m)
            s=source_pos,                           # Source position [x y z] (m)
            L=room,                                 # Room dimensions [x y z] (m)
            reverberation_time=self.reverb_time,    # Reverberation time (s)
            nsample=self.n_sample,                   # Number of output samples
        )

        # Normalise RIR
        for channel_rir in rirs.T:
            max_val = np.max(np.abs(channel_rir))
            channel_rir /= max_val

        # If diffuse, remove early RIR peaks (remove direct and non-diffuse peaks)
        if diffuse:
            for channel_rir in rirs.T:
                # Get peaks
                peaks, _ = signal.find_peaks(channel_rir, distance=100)
                # if self.is_debug:
                #     plt.figure()
                #     plt.plot(channel_rir)
                #     plt.plot(peaks, channel_rir[peaks], 'x')
                #     plt.show()

                # Remove peaks from RIR
                # TODO: CHECK IF REMOVED ENOUGH PEAKS
                min_peak_idx = peaks[10]
                channel_rir[:min_peak_idx] = 0

                # Renormalise RIR
                max_val = np.max(np.abs(channel_rir))
                channel_rir /= max_val

            if self.is_debug:
                plt.figure()
                plt.plot(rirs)
                plt.show()

        # Apply RIR to signal
        source_with_rir = self.apply_rir(rirs, source_audio)

        return source_with_rir

    def generate_config_dict(self, subfolder_name, source_pos, noise_pos, source_name, noise_names, snr=1):
        """
        Generates dict about config for run.

        Args:
            subfolder_name (str): Output subfolder path (contains source and noise names if in real-time).
            source_pos (ndarray): Source positions.
            noise_pos(list[ndarray]): Noise positions (list).
            source_name (str): Name of source audio file used.
            noise_names (list[str]): List of names of noise audio files used.

        Returns:
            Dict with all useful info for run.
        """
        config_dict = dict(
            path=subfolder_name,
            n_channels=self.n_channels,
            microphones=self.receiver_abs,
            room=self.current_room,
            user_pos=self.user_pos.tolist(),
            source_pos=np.around(source_pos, 3).tolist(),
            noise_pos=[np.around(i, 3).tolist() for i in noise_pos],
            source=source_name,
            source_dir=np.around(self.source_direction, 3).tolist(),
            noise=noise_names,
            snr=snr
        )
        return config_dict

    def generate_single_run(self, room=None, source=None, noise=None, number_noises=None):
        """
        Generate a single audio file.

        Args:
            room(list[int, int, int]): Room dimensions to use.
            source (str): Source path to use.
            noise (list[str]): Noise paths to use.
            number_noises (int): Force a number of noises.
        Returns:
            Dictionary containing single audio file with ground truths and configs.
        """
        # Get random room if not given
        if not room:
            random_index = np.random.randint(0, len(self.rooms))
            room = self.rooms[random_index]
        self.set_user_position(room)

        # Get random source if not given
        if source:
            source_path = source
        else:
            random_index = np.random.randint(0, len(self.source_paths))
            source_path = self.source_paths[random_index]
        source_name = source_path.split('\\')[-1].split('.')[0]
        source_audio = self.read_audio_file(source_path)
        source_pos = self.get_random_position(room)

        # Get random noises if not given
        if noise:
            noise_path_list = noise
        else:
            noise_path_list = self.get_random_noise(number_noises)
        noise_pos_list = self.get_random_position(room, source_pos)
        # For each noise get name, RIR and ground truth
        noise_name_list = []
        noise_audio_list = []
        for noise_source_path, noise_pos in zip(noise_path_list, noise_pos_list):
            noise_name_list.append(noise_source_path.split('\\')[-1].split('.')[0])
            noise_audio = self.read_audio_file(noise_source_path)
            noise_audio_list.append(noise_audio)

        # Truncate audio and noises
        source_audio, noise_audio_list = self.truncate_sources(source_audio, noise_audio_list)

        # Get audio RIR
        source_with_rir = self.generate_and_apply_rirs(source_audio, source_pos, room)

        # Get noise RIR
        noise_rir_list = []
        for noise_audio, noise_pos in zip(noise_audio_list, noise_pos_list):
            noise_with_rir = self.generate_and_apply_rirs(noise_audio, noise_pos, room)
            noise_rir_list.append(noise_with_rir)

        # Combine noises
        combined_noise_rir = self.combine_sources(noise_rir_list)

        # Combine source with noises
        audio = [source_with_rir, combined_noise_rir]
        combined_audio = self.combine_sources(audio)

        # Visualize 3D room
        if self.is_debug:
            self.plot_3d_room(source_pos, source_name, noise_pos_list, noise_name_list)

        # Save data to dict
        run_name = f'{source_name}'
        for noise_name in noise_name_list:
            run_name += '_' + noise_name
        config_dict = self.generate_config_dict(run_name, source_pos, noise_pos_list,
                                                source_name, noise_name_list)
        run_dict = dict()
        run_dict['audio'] = combined_audio
        run_dict['target'] = source_audio
        run_dict['speech'] = source_with_rir
        run_dict['noise'] = combined_noise_rir
        run_dict['configs'] = config_dict

        return run_dict

    def generate_dataset(self, save_run):
        """
        Main dataset generator function. Loops over rooms and sources and generates dataset.

        Args:
            save_run (bool): Save dataset to memory or not

        Returns:
            file_count: Number of audio files (subfolders) generated.
        """
        file_count = 0
        # For each room
        for room in self.rooms:
            # TODO: CHANGE ROOM FOR A POLYGON
            # Generate receiver positions from room dimensions
            print(f"Starting for a room of dimensions {room}.")

            # Run through every source
            for source_path in tqdm(self.source_paths, desc="Source Paths used"):
                # Get user random position and source_audio for every source_audio
                self.set_user_position(room)
                source_name = source_path.split('\\')[-1].split('.')[0]  # Get filename for source (before extension)
                source_audio_base = self.read_audio_file(source_path)

                # Run SAMPLES_PER_SPEECH samples per speech clip
                for _ in range(self.sample_per_speech):
                    file_count += 1
                    # Generate source position and copy source_audio
                    source_pos = self.get_random_position(room)
                    source_audio = source_audio_base.copy()

                    # Add varying number of directional noise sources
                    dir_noise_source_paths = self.get_random_noise()
                    dir_noise_pos_list = self.get_random_position(room, source_pos)
                    dir_noise_name_list = []
                    dir_noise_audio_list = []
                    for noise_source_path in dir_noise_source_paths:
                        dir_noise_name_list.append(noise_source_path.split('\\')[-1].split('.')[0])
                        noise_audio = self.read_audio_file(noise_source_path)
                        dir_noise_audio_list.append(noise_audio)

                    # Add varying number of directional noise sources
                    dif_noise_source_paths = self.get_random_noise(diffuse_noise=True)
                    dif_noise_pos_list = self.get_random_position(room, source_pos)
                    dif_noise_name_list = []
                    dif_noise_audio_list = []
                    for noise_source_path in dif_noise_source_paths:
                        dif_noise_name_list.append(noise_source_path.split('\\')[-1].split('.')[0])
                        noise_audio = self.read_audio_file(noise_source_path)
                        dif_noise_audio_list.append(noise_audio)

                    # Truncate noise and source short to shortest length
                    source_audio, dir_noise_audio_list, dif_noise_audio_list = \
                        self.truncate_sources(source_audio, dir_noise_audio_list, dif_noise_audio_list)

                    # Generate source RIR and spectrogram
                    source_with_rir = self.generate_and_apply_rirs(source_audio, source_pos, room)

                    # Calculate rir for directional noises
                    dir_noise_rir_list = []
                    for noise_audio, noise_pos in zip(dir_noise_audio_list, dir_noise_pos_list):
                        noise_with_rir = self.generate_and_apply_rirs(noise_audio, noise_pos, room)
                        dir_noise_rir_list.append(noise_with_rir)

                    # Calculate rir for diffuse noises
                    dif_noise_rir_list = []
                    for noise_audio, noise_pos in zip(dif_noise_audio_list, dif_noise_pos_list):
                        noise_with_rir = self.generate_and_apply_rirs(noise_audio, noise_pos, room, diffuse=True)
                        dif_noise_rir_list.append(noise_with_rir)

                    # Combine noises
                    dir_noise_rir_list.extend(dif_noise_rir_list)
                    dir_noise_pos_list.extend(dif_noise_pos_list)
                    dir_noise_name_list.extend(dif_noise_name_list)
                    combined_noise_rir, _, _ = self.combine_sources(dir_noise_rir_list)

                    # Combine source with noises at a random SNR between limits
                    audio = [source_with_rir.copy(), combined_noise_rir.copy()]
                    snr = np.random.rand()*self.snr_limits[0] + (self.snr_limits[1] - self.snr_limits[0])
                    # snr = 0.1
                    snr = 20 * np.log10(snr)
                    combined_audio, combined_noise_rir, source_with_rir = self.combine_sources(audio, snr)
                    snr = float(10**(snr/20))

                    # Visualize 3D room
                    if self.is_debug:
                        self.plot_3d_room(source_pos, source_name, dir_noise_pos_list, dir_noise_name_list)

                    # Save elements
                    if save_run:
                        subfolder_path = self.save_files(combined_audio, source_with_rir, combined_noise_rir,
                                                         source_audio, source_name, source_pos,
                                                         dir_noise_name_list, dir_noise_pos_list, snr=snr)

                        self.dataset_list.append(subfolder_path)
                        if self.is_debug:
                            print("Created: " + subfolder_path)
                    else:
                        # Generate config dict
                        run_name = f'{source_name}'
                        for noise_name in dir_noise_name_list:
                            run_name += '_' + noise_name
                        config_dict = self.generate_config_dict(run_name, source_pos, dir_noise_pos_list,
                                                                source_name, dir_noise_name_list)

                        # Generate dict to represent 1 element
                        run_dict = dict()
                        run_dict['audio'] = combined_audio
                        run_dict['target'] = source_audio
                        run_dict['speech'] = source_with_rir
                        run_dict['noise'] = combined_noise_rir
                        run_dict['configs'] = config_dict
                        self.dataset_list.append(run_dict)

        return file_count, self.dataset_list
