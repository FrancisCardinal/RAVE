import glob
import os
import yaml
from tqdm import tqdm

import random
from shapely.geometry import Polygon, Point

import audiolib

import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt
import pyroomacoustics as pra

import soundfile as sf
from pyodas.utils import sqrt_hann


SIDE_ID = 0         # X
DEPTH_ID = 1         # Y
HEIGHT_ID = 2         # Z

SAMPLE_RATE = 16000
FRAME_SIZE = 1024
SOUND_MARGIN = 1            # Assume every sound source is margins away from receiver and each other
SOURCE_USER_DISTANCE = 5    # Radius of circle in which source can be
USER_MIN_DISTANCE = 2       # Minimum distance needed between user and walls

MAX_POS_TRIES = 50
TILT_RANGE = 0.25
HEAD_RADIUS = 0.1       # in meters

SHOW_GRAPHS = False

IS_DEBUG = False

class AudioDatasetBuilder:
    """
    Class which handles the generation of the audio dataset through domain randomization of parameters passed through
    dataset_config.yaml and constructor params.

    Args:
        sources_path (str): Path to sources directory. If None, gets folder from config file.
        noises_path (str): Path to noise directory. If None, gets folder from config file.
        output_path (str): Path to output directory.
        noise_count_range (list(int, int)): Range of number of noises.
        speech_noise (bool): Whether to use speech as noise.
        sample_per_speech (int): Number of samples to generate per speech .wav file.
        debug (bool): Run in debug mode.
    """

    user_pos = []
    user_dir = []
    source_direction = []
    current_room = []
    current_room_shape = []
    current_room_size = []
    snr = 1
    receiver_height = 1.5
    receiver_rel = np.array((                   # Receiver (microphone) positions relative to "user" [x, y, z] (m)
                                [-0.1, -0.1, 0],
                                [-0.1, 0.1, 0],
                                [0.1, -0.1, 0],
                                [0.1, 0.1, 0]
                            ))
    rir_max_order = 5
    rir_wall_absorption = 0.85

    def __init__(self, sources_path, noises_path, output_path, noise_count_range,
                 speech_noise, sample_per_speech, debug):

        # Set object values from arguments
        self.dir_noise_count_range = [noise_count_range[0], noise_count_range[1] + 1]
        self.speech_noise = speech_noise
        self.sample_per_speech = sample_per_speech
        self.is_debug = IS_DEBUG

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
        self.room_shapes = self.configs['room_shapes']
        self.room_sizes = self.configs['room_sizes']
        self.banned_noises = self.configs['banned_noises']
        self.diffuse_noises = self.configs['diffuse_noises']
        self.max_diffuse_noises = self.configs['max_diffuse_noises']
        self.snr_limits = self.configs['snr_limits']
        self.wall_absorption_limits = self.configs['wall_absorption_limits']


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
        """
        audio_signal, fs = sf.read(file_path)

        # TODO: Find how to handle if sample rate not at 16 000 (current dataset is all ok)
        if fs != SAMPLE_RATE:
            print(f"ERROR: Sample rate of files ({fs}) do not concord with SAMPLE RATE={SAMPLE_RATE}")
            exit()

        return audio_signal

    @staticmethod
    def apply_rir(rirs, source):
        """
        Method to apply (possibly multichannel) RIRs to a mono-signal.

        Args:
            rirs (ndarray): RIRs generated previously to reflect room, source position and mic positions
            source (ndarray): Mono-channel input signal to be reflected to multiple microphones
        """
        channels = rirs.shape[0]
        frames = len(source['signal'])
        output = np.empty((channels, frames))

        for channel_index in range(channels):
            output[channel_index] = signal.convolve(source['signal'], rirs[channel_index])[:frames]

        source['signal_w_rir'] = output

    @staticmethod
    def truncate_sources(audio_dict):
        """
        Method used to truncate audio sources to smallest one.

        Args:
            audio_dict (dict{str,ndarray,list[int]}): Dictionary containing all audio sources {name, signal, position}.
        """
        # Get length of the shortest audio
        shortest = float('inf')
        for type, source_list in audio_dict.items():
            for source in source_list:
                signal_len = len(source['signal'])
                shortest = signal_len if signal_len < shortest else shortest

        # Truncate all other sources to the shortest length
        for type, source_list in audio_dict.items():
            for source in source_list:
                source['signal'] = source['signal'][:shortest-1]

    @staticmethod
    def generate_spectrogram(signal_x, mono=False, title=''):
        """
        Determines the spectrogram of the input temporal signal through the  Short Term Fourier Transform (STFT).

        Args:
            signal_x (ndarray): Signal on which to get spectrogram.
            mono (bool): Whether the input signal is mono or multichannel.
            title (str): Title used for spectrogram plot (in debug mode only).
        Returns:
            stft_list: List of channels of STFT of input signal x.
        """
        chunk_size = FRAME_SIZE // 2
        window = sqrt_hann(FRAME_SIZE)

        if mono:
            signal_x = [signal_x]

        stft_list = []
        f_log_list = []
        t_list = []
        for channel_idx, channel in enumerate(signal_x):
            # Get stft for every channel
            f, t, stft_x = signal.spectrogram(channel, SAMPLE_RATE, window, FRAME_SIZE, chunk_size)
            t_list.append(t)
            f_log_list.append(np.logspace(0, 4, len(f)))
            stft_log = 10 * np.log10(stft_x)
            stft_list.append(stft_log)

        # Generate plot
        if SHOW_GRAPHS:
            fig, ax = plt.subplots(len(signal_x), constrained_layout=True)
            fig.suptitle(title)
            fig.supylabel('Frequency [Hz]')
            fig.supxlabel('Time [sec]')

            for stft, f, t in zip(stft_list, f_log_list, t_list):
                # Add log and non-log plots
                if SHOW_GRAPHS:
                    if mono:
                        im = ax.pcolormesh(t, f, stft, shading='gouraud')
                        ax.set_yscale('log')
                        fig.colorbar(im, ax=ax)
                    else:
                        im = ax[channel_idx].pcolormesh(t, f, stft, shading='gouraud')
                        ax[channel_idx].set_ylabel(f'Channel_{channel_idx}')
                        ax[channel_idx].set_yscale('log')
                        fig.colorbar(im, ax=ax[channel_idx])

            plt.show()

        return stft_list

    @staticmethod
    def combine_sources(audio_dict, source_types, output_name, noise=False, snr=1):
        """
        Method used to combine audio source with noises.

        Args:
            audio_dict (dict): All audio sources.
            source_types (list[str]): Types of sources to combine.
            output_name (str): Name of output signal to put in dict.
            noise (bool): Check if only noises to add or noise to clean.
            snr (float): Signal to Noise Ratio.
        """
        audio_dict[output_name] = [{'name': '', }]
        audio_dict[output_name][0]['signal_w_rir'] = np.zeros(audio_dict[source_types[0]][0]['signal_w_rir'].shape)
        if not noise:
            # Use audiolib function to add noise to clean with snr
            speech_dict = audio_dict[source_types[0]][0]
            combined_noise_dict = audio_dict[source_types[1]][0]
            audio_dict[output_name][0]['name'] += speech_dict['name'] + '_' + combined_noise_dict['name']

            for c, (speech_channel, noise_channel) in enumerate(zip(speech_dict['signal_w_rir'],
                                                                    combined_noise_dict['signal_w_rir'])):
                speech_snr, noise_snr, combined_snr = audiolib.snr_mixer(speech_channel, noise_channel, snr)
                speech_dict['signal_w_rir'][c] = speech_snr
                combined_noise_dict['signal_w_rir'][c] = noise_snr
                audio_dict[output_name][0]['signal_w_rir'][c] = combined_snr

        else:
            # If noise, just add together
            source_count = 0
            for source_type in source_types:
                for source in audio_dict[source_type]:
                    audio_dict[output_name][0]['name'] += source['name'] + '_'
                    audio_dict[output_name][0]['signal_w_rir'] += source['signal_w_rir']
                    source_count += 1
            audio_dict[output_name][0]['name'] = audio_dict[output_name][0]['name'][:-1]
            audio_dict[output_name][0]['signal_w_rir'] /= source_count

    def plot_scene(self, audio_dict, save_path=None):
        """
        Visualize the virtual room by plotting.

        Args:
            audio_dict (dict): All audio sources.
            save_path (str): Save path for 2D plot.

        """
        # 2D
        # Room
        fig2d, ax = plt.subplots()
        ax.set_xlabel('Side (x)')
        plt.xlim([-5, max(self.current_room_size[0], self.current_room_size[1])+5])
        plt.ylim([-5, max(self.current_room_size[0], self.current_room_size[1])+5])
        ax.set_ylabel('Depth (y)')
        room_corners = self.current_room_shape
        for corner_idx in range(len(room_corners)):
            corner_1 = room_corners[corner_idx]
            corner_2 = room_corners[(corner_idx+1)%len(room_corners)]
            ax.plot([corner_1[0], corner_2[0]], [corner_1[1], corner_2[1]])

        # User
        for mic_pos in self.receiver_abs:
            ax.scatter(mic_pos[SIDE_ID], mic_pos[DEPTH_ID], c='b')
        # ax.text(mic_pos[SIDE_ID], mic_pos[DEPTH_ID], 'User')
        user_dir_point = [self.user_pos[0] + self.user_dir[0],
                          self.user_pos[1] + self.user_dir[1]]
        ax.plot([self.user_pos[0], user_dir_point[0]],
                [self.user_pos[1], user_dir_point[1]])

        # Source
        speech_pos = audio_dict['speech'][0]['position']
        speech_name = audio_dict['speech'][0]['name']
        ax.scatter(speech_pos[SIDE_ID], speech_pos[DEPTH_ID], c='g')
        ax.text(speech_pos[SIDE_ID], speech_pos[DEPTH_ID], speech_name)
        source_circle = plt.Circle((self.user_pos[0]+self.user_dir[0],
                                    self.user_pos[1]+self.user_dir[1]), SOURCE_USER_DISTANCE, color='g', fill=False)
        ax.add_patch(source_circle)

        # Noise
        for dir_noise in audio_dict['dir_noise']:
            dir_noise_pos = dir_noise['position']
            dir_noise_name = dir_noise['name']
            ax.scatter(dir_noise_pos[SIDE_ID], dir_noise_pos[DEPTH_ID], c='m')
            ax.text(dir_noise_pos[SIDE_ID], dir_noise_pos[DEPTH_ID], dir_noise_name)
        for dif_noise in audio_dict['dif_noise']:
            dif_noise_pos = dif_noise['position']
            dif_noise_name = dif_noise['name']
            ax.scatter(dif_noise_pos[SIDE_ID], dif_noise_pos[DEPTH_ID], c='r')
            ax.text(dif_noise_pos[SIDE_ID], dif_noise_pos[DEPTH_ID], dif_noise_name)

        plt.savefig(os.path.join(save_path, 'scene2d.jpg'))

        # 3D
        # Room
        if self.is_debug:
            fig3d, ax = self.current_room.plot(img_order=0)
            ax.set_xlabel('Side (x)')
            ax.set_ylabel('Depth (y)')
            ax.set_zlabel('Height (z)')

            # User
            for mic_pos in self.receiver_abs:
                ax.scatter3D(mic_pos[SIDE_ID], mic_pos[DEPTH_ID], mic_pos[HEIGHT_ID], c='b')
            ax.text(mic_pos[SIDE_ID], mic_pos[DEPTH_ID], mic_pos[HEIGHT_ID], 'User')
            user_dir_point = [self.user_pos[0] + self.user_dir[0],
                              self.user_pos[1] + self.user_dir[1],
                              self.user_pos[2] + self.user_dir[2]]
            ax.plot([self.user_pos[0], user_dir_point[0]],
                    [self.user_pos[1], user_dir_point[1]],
                    [self.user_pos[2], user_dir_point[2]])

            # Source
            ax.scatter3D(speech_pos[SIDE_ID], speech_pos[DEPTH_ID], speech_pos[HEIGHT_ID], c='g')
            ax.text(speech_pos[SIDE_ID], speech_pos[DEPTH_ID], speech_pos[HEIGHT_ID], speech_name)

            # Noise
            for dir_noise in audio_dict['dir_noise']:
                dir_noise_pos = dir_noise['position']
                dir_noise_name = dir_noise['name']
                ax.scatter3D(dir_noise_pos[SIDE_ID], dir_noise_pos[DEPTH_ID], dir_noise_pos[HEIGHT_ID], c='m')
                ax.text(dir_noise_pos[SIDE_ID], dir_noise_pos[DEPTH_ID], dir_noise_pos[HEIGHT_ID], dir_noise_name)
            for dif_noise in audio_dict['dif_noise']:
                dif_noise_pos = dif_noise['position']
                dif_noise_name = dif_noise['name']
                ax.scatter3D(dif_noise_pos[SIDE_ID], dif_noise_pos[DEPTH_ID], dif_noise_pos[HEIGHT_ID], c='r')
                ax.text(dif_noise_pos[SIDE_ID], dif_noise_pos[DEPTH_ID], dif_noise_pos[HEIGHT_ID], dif_noise_name)

        if SHOW_GRAPHS:
            plt.show()

    def generate_random_room(self):
        """
        Generate a random room from room shapes and sizes and creates a pra room with it.

        Returns: PyRoomAcoustics room object.

        """
        # Get random room from room shapes and sizes
        random_room_shape = self.room_shapes[np.random.randint(0, len(self.room_shapes))]
        size_factor = np.random.rand() / 2 + 0.75
        x_factor = np.random.randint(self.room_sizes[0][0], self.room_sizes[0][1])
        y_factor = x_factor * size_factor
        z_factor = np.random.randint(self.room_sizes[2][0], self.room_sizes[2][1])
        self.current_room_size = [x_factor, y_factor, z_factor]

        # Apply size to room shape
        self.current_room_shape = []
        for corner in random_room_shape:
            self.current_room_shape.append([corner[0]*self.current_room_size[0], corner[1]*self.current_room_size[1]])
        corners = np.array(self.current_room_shape).T

        # Generate room from corners and height
        self.rir_wall_absorption = float(np.random.randint(self.wall_absorption_limits[0]*100,
                                                           self.wall_absorption_limits[1]*100)) / 100.
        room = pra.Room.from_corners(corners, fs=SAMPLE_RATE,
                                     max_order=self.rir_max_order,
                                     absorption=self.rir_wall_absorption)
        room.extrude(self.current_room_size[2])
        self.current_room = room

        return room

    def generate_user(self):
        """
        Generate absolute position for user (and for receivers).
        """
        # TODO: GENERATE SPEECH SOURCE ON USER TO REPRESENT USER TALKING?
        # TODO: TEST USER HEAD WITH PYROOMACOUSTICS

        # Get random position and assign x and y wth sound margins (user not stuck on wall)
        self.user_pos = self.get_random_position(user=True)
        while True:
            xy_angle = (np.random.rand() * 2 * math.pi) - math.pi
            z_angle = (np.random.rand() - 0.5) * 2 * TILT_RANGE
            x_dir = (SOURCE_USER_DISTANCE+1) * math.cos(xy_angle+0.5*math.pi)
            y_dir = (SOURCE_USER_DISTANCE+1) * math.sin(xy_angle+0.5*math.pi)
            z_dir = 0.001
            # z_dir = (np.random.rand() - 0.5) * 2 * TILT_RANGE
            if x_dir and y_dir and z_dir:
                break
        self.user_dir = [x_dir, y_dir, z_dir]

        # TODO: IMPLEMENT HEIGHT TILT
        # For every receiver, set x and y by room dimension and add human height as z
        self.receiver_abs = []
        for receiver in self.receiver_rel:
            receiver_center = self.user_pos.copy()
            new_x = receiver[0]*math.cos(xy_angle) - receiver[1]*math.sin(xy_angle)
            new_y = receiver[0]*math.sin(xy_angle) + receiver[1]*math.cos(xy_angle)
            receiver_center[0] += float(new_x)
            receiver_center[1] += float(new_y)
            self.receiver_abs.append(receiver_center)

        # Generate user head walls
        rel_corners = [[-0.1, -0.1], [-0.1, 0.1], [0.1, 0.1], [0.1, -0.1]]
        abs_top_corners = []
        abs_bot_corners = []
        for rel_corner in rel_corners:
            # Rotated axis
            new_x = rel_corner[0] * math.cos(xy_angle) - rel_corner[1] * math.sin(xy_angle)
            new_y = rel_corner[0] * math.sin(xy_angle) + rel_corner[1] * math.cos(xy_angle)
            # Top corners
            corner_top = self.user_pos.copy()
            corner_top[0] += new_x
            corner_top[1] += new_y
            corner_top[2] += 0.1
            abs_top_corners.append(corner_top)
            # Bottom corners
            corner_bot = self.user_pos.copy()
            corner_bot[0] += new_x
            corner_bot[1] += new_y
            corner_bot[2] -= 0.25
            abs_bot_corners.append(corner_bot)

        top_plane = np.array(abs_top_corners).T
        bot_plane = np.array(abs_bot_corners).T
        planes = [top_plane, bot_plane]
        for i in range(4):
            side_plane = np.array([abs_top_corners[i], abs_top_corners[(i+1)%4],
                                   abs_bot_corners[(i+1)%4], abs_bot_corners[i]]).T
            planes.append(side_plane)

        wall_absorption = np.array([0.95])
        scattering = np.array([0])
        for plane in planes:
            self.current_room.walls.append(pra.wall_factory(plane, wall_absorption, scattering, "Head"))

        x_mic_values = [mic[0] for mic in self.receiver_abs]
        y_mic_values = [mic[1] for mic in self.receiver_abs]
        z_mic_values = [mic[2] for mic in self.receiver_abs]
        mic_array = np.array([x_mic_values, y_mic_values, z_mic_values])
        self.current_room.add_microphone_array(pra.MicrophoneArray(mic_array, self.current_room.fs))

    def get_random_position(self, source_pos=None, user=False):
        """
        Get random position inside of polygon composed by PyRoomAcoustics room object walls.

        Args:
            source_pos (list): List of coordinates for source position (if exists).
            user (bool): Whether the random position to generate is for the user.

        Returns: Position composed of shapely.Point with coordinates [x, y, z]. -1 if no position could be made.

        """

        # Get room polygon
        room_poly = Polygon(self.current_room_shape)
        minx, miny, maxx, maxy = room_poly.bounds

        # If source, change min and max to generate close to user
        if not user and not source_pos:
            user_dir_point = [self.user_pos[0] + self.user_dir[0], self.user_pos[1] + self.user_dir[1]]
            minx = max(minx, user_dir_point[0] - SOURCE_USER_DISTANCE)
            miny = max(miny, user_dir_point[1] - SOURCE_USER_DISTANCE)
            maxx = min(maxx, user_dir_point[0] + SOURCE_USER_DISTANCE)
            maxy = min(maxy, user_dir_point[1] + SOURCE_USER_DISTANCE)

        for _ in range(MAX_POS_TRIES):
            # Create random point inside polygon bounds and check if contained
            p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            source_radius = USER_MIN_DISTANCE if user else SOUND_MARGIN
            circle = p.buffer(source_radius)
            if not room_poly.contains(circle):
                continue

            # Set height position for user and/or source
            z = self.receiver_height

            if not user:

                if not source_pos:
                    # If source, check it is contained in circle in front of user
                    dx_user = p.x - user_dir_point[0]
                    dy_user = p.y - user_dir_point[1]
                    is_point_in_user_circle = dx_user**2 + dy_user**2 <= SOURCE_USER_DISTANCE**2
                    if not is_point_in_user_circle:
                        continue

                    self.source_direction = [p.x-self.user_pos[0], p.y-self.user_pos[1], z-self.user_pos[2]]

                else:
                    # Check if noise is not on user
                    dx_user = p.x - self.user_pos[0]
                    dy_user = p.y - self.user_pos[1]
                    is_point_in_user_circle = dx_user * dx_user + dy_user * dy_user <= SOUND_MARGIN * SOUND_MARGIN
                    if is_point_in_user_circle:
                        continue

                    # Check it is not on top of source
                    dx_src = p.x - source_pos[0]
                    dy_src = p.y - source_pos[1]
                    is_point_in_src_circle = dx_src*dx_src + dy_src*dy_src <= SOUND_MARGIN*SOUND_MARGIN
                    if is_point_in_src_circle:
                        continue

                    # Set height position for noise
                    z = np.random.rand() * self.current_room_size[2]

            position = [p.x, p.y, z]
            return position

        return -1

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

    def save_files(self, audio_dict, save_spec=False):
        """
        Save various files needed for dataset (see params).

        Args:
            audio_dict (dict): Contains all info on sound sources.
            save_spec (bool): Whether to save stft (spectrogram) with dataset files.
        Returns:
            subfolder_path (str): String containing path to newly created dataset subfolder.
        """
        # Create subfolder
        subfolder_name = audio_dict['combined_audio'][0]['name']
        noise_quantity = len(audio_dict['dir_noise']) + len(audio_dict['dif_noise'])
        subfolder_path = os.path.join(self.output_subfolder, f'{noise_quantity}', subfolder_name)
        subfolder_index = 1
        if os.path.exists(subfolder_path):
            while os.path.exists(subfolder_path + f'_{subfolder_index}'):
                subfolder_index += 1
            subfolder_path += f'_{subfolder_index}'
        os.makedirs(subfolder_path, exist_ok=True)

        # Save combined audio
        audio_file_name = os.path.join(subfolder_path, 'audio.wav')
        combined_signal = audio_dict['combined_audio'][0]['signal_w_rir']
        sf.write(audio_file_name, combined_signal.T, SAMPLE_RATE)
        if save_spec:
            combined_gt = self.generate_spectrogram(combined_signal, title='Audio')
            audio_gt_name = os.path.join(subfolder_path, 'audio.npz')
            np.savez_compressed(audio_gt_name, combined_gt)

        # Save target
        target_file_name = os.path.join(subfolder_path, 'target.wav')
        target_signal = audio_dict['speech'][0]['signal']
        sf.write(target_file_name, target_signal, SAMPLE_RATE)
        if save_spec:
            # Save source (speech)
            target_gt = self.generate_spectrogram(target_signal, mono=True, title='Target')
            target_gt_name = os.path.join(subfolder_path, 'target.npz')
            np.savez_compressed(target_gt_name, target_gt)

        # Save source (with rir)
        speech_file_name = os.path.join(subfolder_path, 'speech.wav')
        speech_signal = audio_dict['speech'][0]['signal_w_rir']
        sf.write(speech_file_name, speech_signal.T, SAMPLE_RATE)
        if save_spec:
            source_gt = self.generate_spectrogram(speech_signal, title='Speech')
            source_gt_name = os.path.join(subfolder_path, 'speech.npz')
            np.savez_compressed(source_gt_name, source_gt)

        # Save combined noise
        noise_file_name = os.path.join(subfolder_path, 'noise.wav')
        combined_noise = audio_dict['combined_noise'][0]['signal_w_rir']
        sf.write(noise_file_name, combined_noise.T, SAMPLE_RATE)
        if save_spec:
            noise_gt = self.generate_spectrogram(combined_noise, title='Noise')
            noise_gt_name = os.path.join(subfolder_path, 'noise.npz')
            np.savez_compressed(noise_gt_name, noise_gt)

        # Save yaml file with configs
        config_dict_file_name = os.path.join(subfolder_path, 'configs.yaml')
        config_dict = self.generate_config_dict(audio_dict, subfolder_path)
        with open(config_dict_file_name, 'w') as outfile:
            yaml.dump(config_dict, outfile, default_flow_style=None)

        # Visualize and save scene
        self.plot_scene(audio_dict, subfolder_path)

        return subfolder_path

    def generate_and_apply_rirs(self, audio_dict):
        """
        Function that generates Room Impulse Responses (RIRs) to source signal positions and applies them to signals.
        See <https://pyroomacoustics.readthedocs.io/en/pypi-release/>.
        Will have to look later on at
        <https://github.com/DavidDiazGuerra/gpuRIR#simulatetrajectory> for moving sources.

        Args:
            audio_dict (dict{}): Dictionary containing all audio sources and info.
        """
        # TODO: MOVING RIR

        # Generate RIRs
        self.current_room.compute_rir()
        rir_list = self.current_room.rir
        rirs = np.array(rir_list)

        # Normalise RIR
        for channel_idx, channel_rirs in enumerate(rirs):
            for rir in channel_rirs:
                max_val = np.max(np.abs(rir))
                rir /= max_val

        # If diffuse, remove early RIR peaks (remove direct and non-diffuse peaks)
        for rir_channel in rirs:
            for diffuse_rir_idx in range(len(audio_dict['dif_noise'])):
                # Get peaks
                rir = rir_channel[len(audio_dict['speech'])+len(audio_dict['dir_noise'])+diffuse_rir_idx]
                peaks, _ = signal.find_peaks(rir, distance=100)
                if SHOW_GRAPHS:
                    plt.figure()
                    plt.plot(rir)
                    plt.plot(peaks, rir[peaks], 'x')
                    plt.show()

                # Remove peaks from RIR
                min_peak_idx = peaks[len(peaks)//2]
                rir[:min_peak_idx] = 0

                # Renormalise RIR
                max_val = np.max(np.abs(rir))
                rir /= max_val

                if SHOW_GRAPHS:
                    plt.figure()
                    plt.plot(rir)
                    plt.show()

        if SHOW_GRAPHS:
            for channel_rir in rirs:
                plt.figure()
                rir_plot = channel_rir[0].T
                plt.plot(rir_plot)
            plt.show()

        # Apply RIR to signal
        rir_idx = 0
        for source_type, source_lists in audio_dict.items():
            for source in source_lists:
                rir = rirs[:, rir_idx]
                if self.is_debug:
                    source['rir'] = rir
                self.apply_rir(rir, source)

        if SHOW_GRAPHS:
            plt.figure()
            plt.plot(audio_dict['speech'][0]['signal'])
            plt.figure()
            plt.plot(audio_dict['speech'][0]['signal_w_rir'].T)
            plt.show()

    def generate_config_dict(self, audio_dict, subfolder_name):
        """
        Generates dict about config for run.

        Args:
            audio_dict (dict): Dict containing all source information.
            subfolder_name (str): Output subfolder path (contains source and noise names if in real-time).

        Returns:
            Dict with all useful info for run.
        """
        # Get info from dict
        speech_pos = audio_dict['speech'][0]['position']
        speech_name = audio_dict['speech'][0]['name']
        noise_pos = []
        noise_names = []
        for dir_noise in audio_dict['dir_noise']:
            noise_pos.append(dir_noise['position'])
            noise_names.append(dir_noise['name'])
        for dif_noise in audio_dict['dir_noise']:
            noise_pos.append(dif_noise['position'])
            noise_names.append(dif_noise['name'])

        config_dict = dict(
            path=subfolder_name,
            n_channels=self.n_channels,
            microphones=self.receiver_abs,
            room_shape=self.current_room_shape,
            room_size=self.current_room_size,
            user_pos=self.user_pos,
            user_dir=self.user_dir,
            speech_pos=np.around(speech_pos, 3).tolist(),
            noise_pos=[np.around(i, 3).tolist() for i in noise_pos],
            speech=speech_name,
            source_dir=np.around(self.source_direction, 3).tolist(),
            noise=noise_names,
            snr=self.snr,
            rir_order=self.rir_max_order,
            wall_absorption=self.rir_wall_absorption
        )
        return config_dict

    def generate_single_run(self, room=None, source=None, noise=None, number_noises=None):
        """
        DEPRECATED, DO NOT USE UNTIL FIXED. NO USE SO NO DEV WILL BE MADE. TO BE REMOVED IF NOT USED.

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
        self.generate_user(room)

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
        source_with_rir = self.generate_and_apply_rirs(source_audio, room)

        # Get noise RIR
        noise_rir_list = []
        for noise_audio, noise_pos in zip(noise_audio_list, noise_pos_list):
            noise_with_rir = self.generate_and_apply_rirs(noise_audio, room)
            noise_rir_list.append(noise_with_rir)

        # Combine noises
        combined_noise_rir = self.combine_sources(noise_rir_list)

        # Combine source with noises
        audio = [source_with_rir, combined_noise_rir]
        combined_audio = self.combine_sources(audio)

        # Visualize 3D room
        if self.is_debug:
            self.plot_scene(source_pos, source_name, noise_pos_list, noise_name_list, )

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
        print(f"Starting to generate dataset with {self.configs}.")

        # Run through every source
        for source_path in tqdm(self.source_paths, desc="Source Paths used"):
            # Get source_audio for every source file
            source_name = source_path.split('\\')[-1].split('.')[0]  # Get filename for source (before extension)
            source_audio_base = self.read_audio_file(source_path)

            # Run SAMPLES_PER_SPEECH samples per speech clip
            for _ in range(self.sample_per_speech):
                file_count += 1
                audio_source_dict = dict()
                # Get random room and generate user at a position in room
                self.generate_random_room()
                self.generate_user()

                # Generate source position and copy source_audio
                source_pos = self.get_random_position()
                if source_pos == -1:
                    print(f"Source position could not be made with {MAX_POS_TRIES}. Restarting new room.")
                    file_count -= 1
                    continue
                source_audio = source_audio_base.copy()
                audio_source_dict['speech'] = [{'name': source_name, 'signal': source_audio, 'position': source_pos}]

                # Add varying number of directional noise sources
                audio_source_dict['dir_noise'] = []
                dir_noise_source_paths = self.get_random_noise()
                for noise_source_path in dir_noise_source_paths:
                    dir_noise = dict()
                    dir_noise['name'] = noise_source_path.split('\\')[-1].split('.')[0]
                    dir_noise['signal'] = self.read_audio_file(noise_source_path)
                    dir_noise['position'] = self.get_random_position(source_pos)
                    audio_source_dict['dir_noise'].append(dir_noise)

                # Add varying number of directional noise sources
                audio_source_dict['dif_noise'] = []
                dif_noise_source_paths = self.get_random_noise(diffuse_noise=True)
                for noise_source_path in dif_noise_source_paths:
                    dif_noise = dict()
                    dif_noise['name'] = noise_source_path.split('\\')[-1].split('.')[0]
                    dif_noise['signal'] = self.read_audio_file(noise_source_path)
                    dif_noise['position'] = self.get_random_position(source_pos)
                    audio_source_dict['dif_noise'].append(dif_noise)

                # Truncate noise and source short to the shortest length
                self.truncate_sources(audio_source_dict)

                # Add sources to PyRoom and generate RIRs
                for source_list in audio_source_dict.values():
                    for source in source_list:
                        self.current_room.add_source(source['position'], source['signal'])
                self.generate_and_apply_rirs(audio_source_dict)

                # Combine noises
                self.combine_sources(audio_source_dict, ['dir_noise', 'dif_noise'], 'combined_noise', noise=True)

                # Combine source with noises at a random SNR between limits
                # audio = [audio_source_dict['speech'][0]['signal_w_rir'].copy(),
                #          audio_source_dict['combined_noise'][0]['signal_w_rir'].copy()]
                snr = np.random.rand()*self.snr_limits[0] + (self.snr_limits[1] - self.snr_limits[0])
                self.snr = 20 * np.log10(snr)
                self.combine_sources(audio_source_dict, ['speech', 'combined_noise'], 'combined_audio', snr=self.snr)
                self.snr = float(10**(snr/20))

                # Save elements
                if save_run:
                    subfolder_path = self.save_files(audio_source_dict)

                    self.dataset_list.append(subfolder_path)
                    if self.is_debug:
                        print("Created: " + subfolder_path)
                else:
                    # Generate config dict
                    run_name = audio_source_dict['combined_audio'][0]['name']
                    config_dict = self.generate_config_dict(audio_source_dict, run_name)

                    # Generate dict to represent 1 element
                    run_dict = dict()
                    run_dict['audio'] = audio_source_dict['combined_audio'][0]['signal_w_rir']
                    run_dict['target'] = audio_source_dict['speech'][0]['signal']
                    run_dict['speech'] = audio_source_dict['speech'][0]['signal_w_rir']
                    run_dict['noise'] = audio_source_dict['combined_noise'][0]['signal_w_rir']
                    run_dict['configs'] = config_dict
                    self.dataset_list.append(run_dict)

        return file_count, self.dataset_list
