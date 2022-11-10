import os
import yaml

import numpy as np
import matplotlib
import soundfile as sf


class AudioDatasetBuilder:
    """
    Class which handles the generation of the audio dataset through domain randomization of parameters passed through
    dataset_config.yaml and constructor params.

    Args:
        output_path (str): Path to output directory.
        debug (bool): Run in debug mode.
        configs (dict): Dict containing configurations loaded from dataset_config.yaml
    """

    sample_rate = 16000

    user_pos = []
    source_direction = []
    current_room_size = []
    snr = 1

    dif_noise_paths = []
    dir_noise_paths = []

    current_subfolder = None

    # TODO: GENERALIZE MICROPHONE ARRAY
    receiver_rel = np.array(
        (  # Receiver (microphone) geometry relative to "user" [x, y, z] (m)
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

    def __init__(self, output_path, debug, configs):

        # Set object values from arguments
        self.is_debug = debug
        if not self.is_debug:
            matplotlib.use("Agg")

        # Load params/configs
        self.configs = configs
        # Noise configs
        self.banned_noises = self.configs["banned_noises"]
        self.diffuse_noises = self.configs["diffuse_noises"]
        self.max_diffuse_noises = self.configs["max_diffuse_noises"]
        self.dir_noise_count_range = self.configs["dir_noise_count_range"]
        self.dir_noise_count_range = (self.dir_noise_count_range[0], self.dir_noise_count_range[1] + 1)
        self.speech_as_noise = self.configs["speech_as_noise"]
        self.snr_limits = self.configs["snr_limits"]

        # Init
        self.dir_noise_count = self.dir_noise_count_range[0]
        # self.max_source_distance = 5
        self.n_channels = len(self.receiver_rel)

        # Prepare output subfolder
        self.output_subfolder = output_path
        os.makedirs(self.output_subfolder, exist_ok=True)

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
    def read_audio_file(file_path, sample_rate):
        """
        Reads and extracts audio from a file.

        Args:
            file_path (str): Path to audio file
            sample_rate (int): Sample rate at which opened file should be (normally 16000)
        Returns:
            audio_signal (ndarray): Audio read from path of length chunk_size
        """
        audio_signal, fs = sf.read(file_path)

        # if audio_signal.dtype == np.float64:
        #     audio_signal = audio_signal.astype(np.float32)

        if fs != sample_rate:
            print(f"ERROR: Sample rate of files ({fs}) do not concord with SAMPLE RATE={sample_rate}")
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

    def create_subfolder(self, audio_source_dict, output_subfolder):
        """
        Creates subfolder in which to save current dataset element.

        Args:
            audio_source_dict (dict): Dict containing all information on signals (speech and noise)
            output_subfolder (str): String of general output folder in which all dataset is contained.
        Returns:
            String containing current subfolder path.
        """

        subfolder_name = audio_source_dict["speech"][0]["name"]

        for dir_noise in audio_source_dict["dir_noise"]:
            subfolder_name += f"_{dir_noise['name']}"
        for dif_noise in audio_source_dict["dif_noise"]:
            subfolder_name += f"_{dif_noise['name']}"
        noise_quantity = len(audio_source_dict["dir_noise"]) + len(audio_source_dict["dif_noise"])

        if self.is_sim:
            context_str = "reverb" if self.is_reverb else "no_reverb"
        else:
            context_str = os.path.join(self.current_room_size, 'pos_'+self.user_pos)
        subfolder_path = os.path.join(output_subfolder, context_str, f"{noise_quantity}", subfolder_name)

        subfolder_index = 1
        if os.path.exists(subfolder_path):
            while os.path.exists(subfolder_path + f"_{subfolder_index}"):
                subfolder_index += 1
            subfolder_path += f"_{subfolder_index}"
        os.makedirs(subfolder_path, exist_ok=True)

        return subfolder_path

    def generate_config_dict(self, audio_dict, subfolder_name):
        """
        Generates dict to save run configs.

        Args:
            audio_dict (dict): Dict containing all source information.
            subfolder_name (str): Output subfolder path (contains source and noise names if in real-time).

        Returns:
            Dict with all useful info for run.
        """
        # Get audio names from dict
        speech_name = audio_dict["speech"][0]["name"]
        noise_names = []
        for dir_noise in audio_dict["dir_noise"]:
            noise_names.append(dir_noise["name"])
        for dif_noise in audio_dict["dif_noise"]:
            noise_names.append(dif_noise["name"])

        if self.is_sim:
            # Get audio positions from dict
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
                wall_absorption=self.wall_absorption_limits,
            )
        else:
            config_dict = dict(
                path=subfolder_name,
                n_channels=self.n_channels,
                mic_rel=self.receiver_rel.tolist(),
                room_size=self.current_room_size,
                user_pos=self.user_pos,
                speech=speech_name,
                source_dir=np.around(self.source_direction, 3).tolist(),
                noise=noise_names,
                snr=self.snr
            )
        return config_dict
