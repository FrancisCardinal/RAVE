import os
import yaml

import itertools

import numpy as np
import matplotlib

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

    # Same
    @staticmethod
    def combine_sources(audio_dict, source_types, output_name, dict_str, noise=False, snr=1):
        """
        Method used to combine audio source with noises.

        Args:
            audio_dict (dict): All audio sources.
            source_types (list[str]): Types of sources to combine.
            output_name (str): Name of output signal to put in dict.
            dict_str (str): Name to use to refer to output signal in dict ('signal' or 'signal_w_rir').
            noise (bool): Check if only noises to add or noise to clean.
            snr (float): Signal to Noise Ratio (in amplitude).
        """

        # TODO: REFACTOR, maybe use pydub for all combinations?

        audio_dict[output_name] = [
            {
                "name": "",
            }
        ]
        if not noise:
            snr_db = 20 * np.log10(snr)

            speech_db = audio_dict['speech'][0]['audio_segment'].dBFS
            noise_db = audio_dict['combined_noise'][0]['audio_segment'].dBFS
            adjust_snr = speech_db - noise_db - snr_db

            audio_dict['combined_noise'][0]['audio_segment'] = \
                audio_dict['combined_noise'][0]['audio_segment'] + adjust_snr

        audio_dict[output_name][0]['audio_segment'] = AudioSegment.silent(duration=10000)
        for source_type in source_types:
            for source in audio_dict[source_type]:
                audio_dict[output_name][0]["name"] += source["name"] + "_"
                audio_dict[output_name][0]['audio_segment'] = \
                    audio_dict[output_name][0]['audio_segment'].overlay(source['audio_segment'])
        # TODO: Check if need to reduce sound when adding combinations

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

    # Almost same
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

    # Same
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

    # Same
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
        combined_signal = audio_dict["combined_audio"][0]['audio_segment']
        combined_signal.export(audio_file_name, format='wav')
        if save_spec:
            combined_gt = self.generate_spectrogram(combined_signal, title="Audio")
            audio_gt_name = os.path.join(self.current_subfolder, "audio.npz")
            np.savez_compressed(audio_gt_name, combined_gt)

        # Save target
        target_file_name = os.path.join(self.current_subfolder, "target.wav")
        target_signal = audio_dict["speech"][0]['audio_segment']
        target_signal.export(target_file_name, format='wav')
        if save_spec:
            # Save source (speech)
            target_gt = self.generate_spectrogram(target_signal, mono=True, title="Target")
            target_gt_name = os.path.join(self.current_subfolder, "target.npz")
            np.savez_compressed(target_gt_name, target_gt)

        # Save source (with rir)
        speech_file_name = os.path.join(self.current_subfolder, "speech.wav")
        speech_signal = audio_dict["speech"][0]['audio_segment']
        speech_signal.export(speech_file_name, format='wav')
        if save_spec:
            source_gt = self.generate_spectrogram(speech_signal, title="Speech")
            source_gt_name = os.path.join(self.current_subfolder, "speech.npz")
            np.savez_compressed(source_gt_name, source_gt)

        # Save combined noise
        noise_file_name = os.path.join(self.current_subfolder, "noise.wav")
        combined_noise = audio_dict["combined_noise"][0]['audio_segment']
        combined_noise.export(noise_file_name, format='wav')
        if save_spec:
            noise_gt = self.generate_spectrogram(combined_noise, title="Noise")
            noise_gt_name = os.path.join(self.current_subfolder, "noise.npz")
            np.savez_compressed(noise_gt_name, noise_gt)

        # Save yaml file with configs
        config_dict_file_name = os.path.join(self.current_subfolder, "configs.yaml")
        config_dict = self.generate_config_dict(audio_dict, self.current_subfolder)
        with open(config_dict_file_name, "w") as outfile:
            yaml.dump(config_dict, outfile, default_flow_style=None)

        return self.current_subfolder

    # Almost same
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

    def generate_real_dataset(self, source_paths, save_run):
        """
        Main dataset generator function (for real data). Takes all data inside a room folder and mix-and-matches.

        Args:
            source_paths (dict): Dict containing name and paths for {room, speech, noise, other_speech}.
            save_run (bool): Save dataset to memory or not.

        Returns:
            file_count: Number of audio files (subfolders) generated.
        """

        # Split dict
        source_path = source_paths['speech']
        noise_paths = source_paths['noise']
        other_speech_paths = source_paths['other_speech']
        configs = source_paths['configs']

        # Configs
        self.current_room_size = configs['room']
        self.user_pos = configs['user_pos']
        self.source_direction = AudioDatasetBuilder.parse_real_dir(configs['location'])

        self.audio_dict_out_signal_str = 'signal'

        # Get Load audio
        source_name = configs['sound']      # Get filename for source (before extension)
        source_audio_base = AudioSegment.from_wav(source_path)

        # Add speech to noises if needed
        if self.speech_as_noise:
            noise_paths.extend(other_speech_paths)

        # Get every combination of noises possible within filtered noises
        # TODO: Check if we should remove noises if recorded at the same location between themselves?
        noise_source_paths_combinations = []
        for i in range(self.dir_noise_count_range[0], self.dir_noise_count_range[1]):
            combinations = itertools.combinations(noise_paths, i)
            noise_source_paths_combinations.extend(combinations)

        samples_created = 0
        total_snr = 0
        for noise_source_paths in noise_source_paths_combinations:
            audio_source_dict = dict()

            # Copy source_audio and put in list
            source_audio = source_audio_base
            audio_source_dict["speech"] = [{"name": source_name, "audio_segment": source_audio}]

            # Add varying number of directional noise sources
            audio_source_dict["dir_noise"] = []
            for noise_source_path in noise_source_paths:
                noise = dict()
                noise["name"] = os.path.split(noise_source_path)[0].split(os.path.sep)[-1]
                noise["audio_segment"] = AudioSegment.from_wav(noise_source_path)
                audio_source_dict["dir_noise"].append(noise)

            # Add empty dif_noise (to be removed)
            audio_source_dict["dif_noise"] = []

            # Create subfolder
            self.current_subfolder = self.create_subfolder(audio_source_dict, self.output_subfolder)

            # Combine noises
            self.combine_sources(audio_source_dict, ["dir_noise"], "combined_noise",
                                 self.audio_dict_out_signal_str, noise=True)

            # Combine source with noises at a random SNR between limits
            # TODO: CHECK SNR IF IT WORKS CORRECTLY
            self.snr = np.random.rand() * (self.snr_limits[1] - self.snr_limits[0]) + self.snr_limits[0]
            if self.is_debug: total_snr += self.snr
            self.combine_sources(audio_source_dict, ["speech", "combined_noise"], "combined_audio",
                                 self.audio_dict_out_signal_str, snr=self.snr)

            # Save elements
            if save_run:
                subfolder_path = self.save_files(audio_source_dict)

                self.dataset_list.append(subfolder_path)
                if self.is_debug:
                    print("Created: " + subfolder_path)
            else:
                # Generate config dict
                run_name = audio_source_dict["combined_audio"][0]["name"]
                config_dict = self.generate_config_dict(audio_source_dict, run_name)

                # Generate dict to represent 1 element
                run_dict = dict()
                run_dict["audio"] = audio_source_dict["combined_audio"][0][self.audio_dict_out_signal_str]
                run_dict["target"] = audio_source_dict["speech"][0]["signal"]
                run_dict["speech"] = audio_source_dict["speech"][0][self.audio_dict_out_signal_str]
                run_dict["noise"] = audio_source_dict["combined_noise"][0][self.audio_dict_out_signal_str]
                run_dict["configs"] = config_dict
                self.dataset_list.append(run_dict)

            # Increase counters
            samples_created += 1

        if self.is_debug:
            print(f'Mean SNR: {total_snr/samples_created}')

        return samples_created, self.dataset_list
