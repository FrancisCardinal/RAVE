import os
import itertools
import numpy as np
import yaml

from pydub import AudioSegment

from .AudioDatasetBuilder import AudioDatasetBuilder


class AudioDatasetBuilderReal(AudioDatasetBuilder):
    """
    Class which handles the generation of the audio dataset through domain randomization of parameters passed through
    dataset_config.yaml and constructor params.

    Args:
        output_path (str): Path to output directory.
        debug (bool): Run in debug mode.
        configs (dict): Dict containing configurations loaded from dataset_config.yaml
    """

    source_direction = []

    def __init__(self, output_path, debug, configs):

        self.is_sim = False
        super().__init__(output_path, debug, configs)

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

    @staticmethod
    def combine_sources(audio_dict, source_types, output_name, noise=False, snr=1):
        """
        Method used to combine audio source with noises.

        Args:
            audio_dict (dict): All audio sources.
            source_types (list[str]): Types of sources to combine.
            output_name (str): Name to use to output combined signal in audio_dict.
            noise (bool): Check if only noises to add or noise to clean.
            snr (float): Signal to Noise Ratio (in amplitude).
        """

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

    def save_files(self, audio_dict):
        """
        Save various files needed for dataset (see params).

        Args:
            audio_dict (dict): Contains all info on sound sources.
        Returns:
            subfolder_path (str): String containing path to newly created dataset subfolder.
        """

        # Save combined audio
        audio_file_name = os.path.join(self.current_subfolder, "audio.wav")
        combined_signal = audio_dict["combined_audio"][0]['audio_segment']
        combined_signal.export(audio_file_name, format='wav')

        # Save target (mono)
        target_file_name = os.path.join(self.current_subfolder, "target.wav")
        target_signal = audio_dict["speech"][0]["audio_segment"]
        target_signal.export(target_file_name, format='wav')

        # Save source (multi)
        speech_file_name = os.path.join(self.current_subfolder, "speech.wav")
        speech_signal = audio_dict["speech"][0]['audio_segment']
        speech_signal.export(speech_file_name, format='wav')

        # Save combined noise
        noise_file_name = os.path.join(self.current_subfolder, "noise.wav")
        combined_noise = audio_dict["combined_noise"][0]['audio_segment']
        combined_noise.export(noise_file_name, format='wav')

        # Save yaml file with configs
        config_dict_file_name = os.path.join(self.current_subfolder, "configs.yaml")
        config_dict = self.generate_config_dict(audio_dict, self.current_subfolder)
        with open(config_dict_file_name, "w") as outfile:
            yaml.dump(config_dict, outfile, default_flow_style=None)

        return self.current_subfolder

    def generate_dataset(self, source_paths, save_run):
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
        self.source_direction = self.parse_real_dir(configs['location'])

        # Get Load audio
        source_name = configs['sound']      # Get filename for source (before extension)
        source_audio_base = AudioSegment.from_wav(source_path)

        # Add speech to noises if needed
        if self.speech_as_noise:
            noise_paths.extend(other_speech_paths)

        # Get every combination of noises possible within filtered noises
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
            self.combine_sources(audio_source_dict, ["dir_noise"], "combined_noise", noise=True)

            # Combine source with noises at a random SNR between limits
            # TODO: CHECK SNR IF IT WORKS CORRECTLY
            self.snr = np.random.rand() * (self.snr_limits[1] - self.snr_limits[0]) + self.snr_limits[0]
            if self.is_debug: total_snr += self.snr
            self.combine_sources(audio_source_dict, ["speech", "combined_noise"], "combined_audio", snr=self.snr)

            # Save elements
            if save_run:
                subfolder_path = self.save_files(audio_source_dict)
                if self.is_debug:
                    print("Created: " + subfolder_path)

            # Increase counters
            samples_created += 1

        if self.is_debug:
            print(f'Mean SNR: {total_snr/samples_created}')

        return samples_created
