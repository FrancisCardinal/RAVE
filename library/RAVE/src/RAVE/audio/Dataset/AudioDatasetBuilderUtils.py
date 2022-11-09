import glob
import os
import yaml

import random
from shapely.geometry import Polygon, Point

import itertools

import numpy as np
import math
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
import pyroomacoustics as pra

import soundfile as sf
from pyodas.utils import sqrt_hann

from pydub import AudioSegment


SIDE_ID = 0  # X
DEPTH_ID = 1  # Y
HEIGHT_ID = 2  # Z

SAMPLE_RATE = 16000
FRAME_SIZE = 1024
SOUND_MARGIN = 1  # Assume every sound source is margins away from receiver and each other
SOURCE_USER_DISTANCE = 5  # Radius of circle in which source can be
USER_MIN_DISTANCE = 2  # Minimum distance needed between user and walls

MAX_POS_TRIES = 50
TILT_RANGE = 0.25
HEAD_RADIUS = 0.1  # in meters

SHOW_GRAPHS = False
SAVE_RIR = True

IS_DEBUG = False


class AudioDatasetBuilder:
    """
    Class which handles the generation of the audio dataset through domain randomization of parameters passed through
    dataset_config.yaml and constructor params.

    Args:
        output_path (str): Path to output directory.
        debug (bool): Run in debug mode.
        configs (dict): Dict containing configurations loaded from dataset_config.yaml
    """

    user_pos = []
    user_dir = []
    xy_angle = 0
    source_direction = []
    current_room = []
    current_room_shape = []
    current_room_size = []
    snr = 1
    receiver_height = 1.5

    # TODO: GENERALIZE MICROPHONE ARRAY
    receiver_rel = np.array(
        (  # Receiver (microphone) positions relative to "user" [x, y, z] (m)
            [-0.07055, 0, 0],
            [-0.07055, 0.0381, 0],
            [-0.05715, 0.0618, 0],
            [-0.01905, 0.0618, 0],
            [0.01905, 0.0618, 0],
            [0.05715, 0.0618, 0],
            [0.07055, 0.0381, 0],
            [0.07055, 0, 0]
        )

    )

    def __init__(self, output_path, debug, configs, sim=True):

        # Set object values from arguments
        self.is_debug = debug
        self.is_sim = sim

        if not self.is_debug:
            matplotlib.use("Agg")

        # TODO: Split configs between real and sim
        # Load params/configs
        self.configs = configs
        # Room configs
        self.room_shapes = self.configs["room_shapes"]
        self.reverb_room_shapes = self.configs["reverb_room_shapes"]
        self.room_sizes = self.configs["room_sizes"]
        # Noise configs
        self.banned_noises = self.configs["banned_noises"]
        self.diffuse_noises = self.configs["diffuse_noises"]
        self.max_diffuse_noises = self.configs["max_diffuse_noises"]
        self.dir_noise_count_range = self.configs["dir_noise_count_range"]
        self.dir_noise_count_range = (self.dir_noise_count_range[0], self.dir_noise_count_range[1] + 1)
        self.sample_per_speech = self.configs["sample_per_speech"]
        self.speech_as_noise = self.configs["speech_as_noise"]
        self.snr_limits = self.configs["snr_limits"]
        # RIR configs
        self.wall_absorption_limits = self.configs["wall_absorption_limits"]
        self.rir_reflexion_order = self.configs["rir_reflexion_order"]
        self.air_absorption = self.configs["air_absorption"]
        self.air_humidity = self.configs["air_humidity"]

        self.receiver_abs = None
        self.dir_noise_count = self.dir_noise_count_range[0]
        self.speech_noise_start = 0
        self.max_source_distance = 5
        self.n_channels = len(self.receiver_rel)

        self.is_reverb = None
        self.noise_paths = None
        self.dif_noise_paths = []
        self.dir_noise_paths = []
        self.room = None

        # Dict strings to vary between real and sim datasets
        self.audio_dict_out_signal_str = ''

        # Prepare output subfolder
        self.output_subfolder = output_path
        os.makedirs(self.output_subfolder, exist_ok=True)
        self.current_subfolder = None

        # Prepare lists for run-time generation
        self.dataset_list = []

    @staticmethod
    def load_configs(config_path):
        """
        Load configs file for dataset builder.

        Args:
            config_path (str): Path where configs file is located.
        """
        with open(config_path, "r") as stream:
            try:
                configs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return configs

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
            print(f"Use sample_adjustment python file to adjust sample rate.")
            exit()

        return audio_signal

    @staticmethod
    def truncate_sources(audio_dict):
        """
        Method used to truncate audio sources to smallest one.

        Args:
            audio_dict (dict{str,ndarray,list[int]}): Dictionary containing all audio sources {name, signal, position}.
        """
        # Get length of the shortest audio
        shortest = float("inf")
        for type, source_list in audio_dict.items():
            for source in source_list:
                signal_len = len(source["signal"])
                shortest = signal_len if signal_len < shortest else shortest

        # Truncate all other sources to the shortest length
        for type, source_list in audio_dict.items():
            for source in source_list:
                source["signal"] = source["signal"][: shortest - 1]

    @staticmethod
    def generate_spectrogram(signal_x, mono=False, title=""):
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
            fig.supylabel("Frequency [Hz]")
            fig.supxlabel("Time [sec]")

            for stft, f, t in zip(stft_list, f_log_list, t_list):
                # Add log and non-log plots
                if SHOW_GRAPHS:
                    if mono:
                        im = ax.pcolormesh(t, f, stft, shading="gouraud")
                        ax.set_yscale("log")
                        fig.colorbar(im, ax=ax)
                    else:
                        im = ax[channel_idx].pcolormesh(t, f, stft, shading="gouraud")
                        ax[channel_idx].set_ylabel(f"Channel_{channel_idx}")
                        ax[channel_idx].set_yscale("log")
                        fig.colorbar(im, ax=ax[channel_idx])

            plt.show()

        return stft_list

    @staticmethod
    def parse_real_dir(location_str):
        """
        Transform the location string from recorded samples to coords.
        Args:
            location_str: String containing speech direction (form (y, x, z), 170_m80_m10) in cm).

        Returns:
            List of coordinates to speech direction in m.
        """

        coord_str_list = location_str.split('_')

        coord_list = []
        for coord_str in coord_str_list:
            factor = 1
            if coord_str[0] == 'm':
                factor = -1
                coord_str = coord_str[1:]
            coord_int = int(coord_str) * factor / 100   # * negative factor / cm to m
            coord_list.append(coord_int)

        return coord_list

    def create_subfolder(self, audio_source_dict, output_subfolder):
        """
        Creates subfolder in which to save current dataset element.

        Args:
            audio_source_dict (dict): Dict containing all information on signals (speech and noise)
            output_subfolder (str): String of general output folder in which all dataset is contained.
        Returns:
            String containing current subfolder path.
        """

        #TODO: Refactor

        subfolder_name = audio_source_dict["speech"][0]["name"]
        for dir_noise in audio_source_dict["dir_noise"]:
            subfolder_name += f"_{dir_noise['name']}"
        for dif_noise in audio_source_dict["dif_noise"]:
            subfolder_name += f"_{dif_noise['name']}"
        noise_quantity = len(audio_source_dict["dir_noise"]) + len(audio_source_dict["dif_noise"])
        context_str = os.path.join(self.current_room_size, 'pos_'+self.user_pos)
        subfolder_path = os.path.join(output_subfolder, context_str, f"{noise_quantity}", subfolder_name)
        subfolder_index = 1
        if os.path.exists(subfolder_path):
            while os.path.exists(subfolder_path + f"_{subfolder_index}"):
                subfolder_index += 1
            subfolder_path += f"_{subfolder_index}"
        os.makedirs(subfolder_path, exist_ok=True)

        return subfolder_path

    # def get_random_position(self, source_pos=None, user=False):
    #     """
    #     Get random position inside of polygon composed by PyRoomAcoustics room object walls.
    #
    #     Args:
    #         source_pos (list): List of coordinates for source position (if exists).
    #         user (bool): Whether the random position to generate is for the user.
    #
    #     Returns: Position composed of shapely.Point with coordinates [x, y, z]. -1 if no position could be made.
    #
    #     """
    #     # Get room polygon
    #     room_poly = Polygon(self.current_room_shape)
    #     minx, miny, maxx, maxy = room_poly.bounds
    #
    #     # If source, change min and max to generate in user front circle
    #     if not user and not source_pos:
    #         user_dir_point = [self.user_pos[0] + self.user_dir[0], self.user_pos[1] + self.user_dir[1]]
    #         minx = max(minx, user_dir_point[0] - SOURCE_USER_DISTANCE)
    #         miny = max(miny, user_dir_point[1] - SOURCE_USER_DISTANCE)
    #         maxx = min(maxx, user_dir_point[0] + SOURCE_USER_DISTANCE)
    #         maxy = min(maxy, user_dir_point[1] + SOURCE_USER_DISTANCE)
    #
    #     for _ in range(MAX_POS_TRIES):
    #         # Create random point inside polygon bounds and check if contained
    #         p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
    #         source_radius = USER_MIN_DISTANCE if user else SOUND_MARGIN
    #         circle = p.buffer(source_radius)
    #         if not room_poly.contains(circle):
    #             continue
    #
    #         # Set height position for user and/or source
    #         z = self.receiver_height
    #
    #         if not user:
    #             if not source_pos:
    #                 # If source, check it is contained in circle in front of user
    #                 dx_user = p.x - user_dir_point[0]
    #                 dy_user = p.y - user_dir_point[1]
    #                 is_point_in_user_circle = dx_user**2 + dy_user**2 <= SOURCE_USER_DISTANCE**2
    #                 if not is_point_in_user_circle:
    #                     continue
    #
    #                 # Calculate source direction based on user direction
    #                 # TODO: add tilt
    #                 coords = [
    #                     p.x - self.user_pos[SIDE_ID],
    #                     p.y - self.user_pos[DEPTH_ID],
    #                     z - self.user_pos[HEIGHT_ID],
    #                 ]
    #                 new_x, new_y, new_z = self.rotate_coords(coords, self.xy_angle, inverse=True)
    #                 front_facing_angle = 0.5 * math.pi
    #                 new_x, new_y, new_z = self.rotate_coords([new_x, new_y, new_z], front_facing_angle, inverse=False)
    #                 self.source_direction = [new_x, new_y, new_z]
    #
    #             else:
    #                 # Check if noise is not on user
    #                 dx_user = p.x - self.user_pos[0]
    #                 dy_user = p.y - self.user_pos[1]
    #                 is_point_in_user_circle = dx_user * dx_user + dy_user * dy_user <= SOUND_MARGIN * SOUND_MARGIN
    #                 if is_point_in_user_circle:
    #                     continue
    #
    #                 # Check it is not on top of source
    #                 dx_src = p.x - source_pos[0]
    #                 dy_src = p.y - source_pos[1]
    #                 is_point_in_src_circle = dx_src**2 + dy_src**2 <= SOUND_MARGIN**2
    #                 if is_point_in_src_circle:
    #                     continue
    #
    #                 # Set height position for noise
    #                 z = np.random.rand() * self.current_room_size[2]
    #
    #         position = [p.x, p.y, z]
    #         return position
    #
    #     return -1

    def get_random_noise(self, number_noises=None, diffuse_noise=False, source_path=None, sn_count=-1):
        """
        Gets random noises to be added to audio clip.

        Args:
            number_noises (int): Number of noises to use (if specified, overrides self.noise_count).
            diffuse_noise (bool): Bool to use diffuse or directional noise.
            source_path (str): String of speech path.
            sn_count (int): Speech_noise count in last try (-1 if nothing to force).
        Returns:
            List of paths to noises (and/or speech) to use.
        """

        if diffuse_noise:
            random_indices = np.random.randint(0, len(self.dif_noise_paths), self.max_diffuse_noises)
            noise_path_list = [self.dif_noise_paths[i] for i in random_indices]
            speech_noise_count = None
        else:
            # Set noise count for this round
            if number_noises:
                self.dir_noise_count = number_noises
            else:
                if sn_count == -1:      # Only increase number of noises if not having to restart the last run
                    self.dir_noise_count -= self.dir_noise_count_range[0]
                    self.dir_noise_count += 1
                    dir_count_range = self.dir_noise_count_range
                    self.dir_noise_count = self.dir_noise_count % (dir_count_range[1] - dir_count_range[0])
                    self.dir_noise_count += self.dir_noise_count_range[0]

            # Get random indices and return items in new list
            temp_noise_paths = self.dir_noise_paths.copy()
            if source_path in temp_noise_paths:
                temp_noise_paths.remove(source_path)

            # If previous run breaks, remake run with same quantity of speech noise
            if sn_count != -1 and self.speech_as_noise:
                random_speech_indices = np.random.randint(self.speech_noise_start, len(temp_noise_paths), sn_count)
                random_noise_indices = np.random.randint(0, self.speech_noise_start, self.dir_noise_count - sn_count)
                random_indices = np.concatenate((random_speech_indices, random_noise_indices))
            else:
                random_indices = np.random.randint(0, len(temp_noise_paths), self.dir_noise_count)

            noise_path_list = []
            speech_noise_count = 0
            for i in random_indices:
                selected_noise = temp_noise_paths[i]
                if 'clean' in selected_noise: speech_noise_count += 1
            noise_path_list = [temp_noise_paths[int(i)] for i in random_indices]

        return noise_path_list, speech_noise_count

    def save_files(self, audio_dict, save_spec=False):
        """
        Save various files needed for dataset (see params).

        Args:
            audio_dict (dict): Contains all info on sound sources.
            save_spec (bool): Whether to save stft (spectrogram) with dataset files.
        Returns:
            subfolder_path (str): String containing path to newly created dataset subfolder.
        """

        # Save combined audio
        audio_file_name = os.path.join(self.current_subfolder, "audio.wav")
        combined_signal = audio_dict["combined_audio"][0][self.audio_dict_out_signal_str]
        if self.is_sim:
            sf.write(audio_file_name, combined_signal, SAMPLE_RATE)
            # sf.write(audio_file_name, combined_signal.T, SAMPLE_RATE)
        else:
            combined_signal.export(audio_file_name, format='wav')
        if save_spec:
            combined_gt = self.generate_spectrogram(combined_signal, title="Audio")
            audio_gt_name = os.path.join(self.current_subfolder, "audio.npz")
            np.savez_compressed(audio_gt_name, combined_gt)

        # Save target
        target_file_name = os.path.join(self.current_subfolder, "target.wav")
        target_signal = audio_dict["speech"][0]["signal"]
        if self.is_sim:
            sf.write(target_file_name, target_signal, SAMPLE_RATE)
        else:
            target_signal.export(target_file_name, format='wav')
        if save_spec:
            # Save source (speech)
            target_gt = self.generate_spectrogram(target_signal, mono=True, title="Target")
            target_gt_name = os.path.join(self.current_subfolder, "target.npz")
            np.savez_compressed(target_gt_name, target_gt)

        # Save source (with rir)
        speech_file_name = os.path.join(self.current_subfolder, "speech.wav")
        speech_signal = audio_dict["speech"][0][self.audio_dict_out_signal_str]
        if self.is_sim:
            sf.write(speech_file_name, speech_signal.T, SAMPLE_RATE)
            # sf.write(speech_file_name, speech_signal, SAMPLE_RATE)
        else:
            speech_signal.export(speech_file_name, format='wav')
        if save_spec:
            source_gt = self.generate_spectrogram(speech_signal, title="Speech")
            source_gt_name = os.path.join(self.current_subfolder, "speech.npz")
            np.savez_compressed(source_gt_name, source_gt)

        # Save combined noise
        noise_file_name = os.path.join(self.current_subfolder, "noise.wav")
        combined_noise = audio_dict["combined_noise"][0][self.audio_dict_out_signal_str]
        if self.is_sim:
            combined_noise.export(noise_file_name, format='wav')
        else:
            sf.write(noise_file_name, combined_noise, SAMPLE_RATE)
            # sf.write(noise_file_name, combined_noise.T, SAMPLE_RATE)
        if save_spec:
            noise_gt = self.generate_spectrogram(combined_noise, title="Noise")
            noise_gt_name = os.path.join(self.current_subfolder, "noise.npz")
            np.savez_compressed(noise_gt_name, noise_gt)

        # Save yaml file with configs
        config_dict_file_name = os.path.join(self.current_subfolder, "configs.yaml")
        config_dict = self.generate_config_dict(audio_dict, self.current_subfolder)
        with open(config_dict_file_name, "w") as outfile:
            yaml.dump(config_dict, outfile, default_flow_style=None)

        # Visualize and save scene
        if self.is_sim:
            self.plot_scene(audio_dict, self.current_subfolder)

        return self.current_subfolder

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
        speech_name = audio_dict["speech"][0]["name"]
        noise_names = []
        for dir_noise in audio_dict["dir_noise"]:
            noise_names.append(dir_noise["name"])
        for dif_noise in audio_dict["dif_noise"]:
            noise_names.append(dif_noise["name"])

        if self.is_sim:
            speech_pos = audio_dict["speech"][0]["position"]
            noise_pos = []
            for dir_noise in audio_dict["dir_noise"]:
                noise_pos.append(dir_noise["position"])
            for dif_noise in audio_dict["dif_noise"]:
                noise_pos.append(dif_noise["position"])

            config_dict = dict(
                path=subfolder_name,
                n_channels=self.n_channels,
                mic_rel=self.receiver_rel.tolist(),
                mic_abs=self.receiver_abs,
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
                rir_reflexion_order=self.rir_reflexion_order,
                wall_absorption=self.rir_wall_absorption,
            )
        else:
            config_dict = dict(
                path=subfolder_name,
                n_channels=self.n_channels,
                mic_rel=self.receiver_rel.tolist(),
                # mic_abs=self.receiver_abs,
                # room_shape=self.current_room_shape,
                room_size=self.current_room_size,
                user_pos=self.user_pos,
                # user_dir=self.user_dir,
                # speech_pos=np.around(speech_pos, 3).tolist(),
                # noise_pos=[np.around(i, 3).tolist() for i in noise_pos],
                speech=speech_name,
                source_dir=np.around(self.source_direction, 3).tolist(),
                noise=noise_names,
                snr=self.snr,
                # rir_reflexion_order=self.rir_reflexion_order,
                # wall_absorption=self.rir_wall_absorption,
            )
        return config_dict
