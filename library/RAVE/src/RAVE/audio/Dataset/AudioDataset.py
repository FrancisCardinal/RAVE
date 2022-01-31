import torch

from tkinter import filedialog
import glob
import os
import random

from audio import AudioDatasetBuilder

MINIMUM_DATASET_LENGTH = 500

class AudioDataset(torch.utils.data.Dataset):
    """
    Class handling usage of the audio dataset
    """

    def __init__(
        self,
        dataset_path=None,
        data_split=None,
        is_debug=False
    ):

        # Set up dataset folder (if none, open tkinter to choose)
        self.dataset_path = dataset_path
        if not self.dataset_path:
            self.dataset_path = filedialog.askdirectory(title="Choose a dataset folder")

        self.is_debug = is_debug

        # Check if dataset_path exists
        self.dataset_item_paths = []
        self.get_dataset(self.dataset_path)

        for path in glob.glob(f'{dataset_path}/*/**/', recursive=True):
            print(path)

        # Split dataset into subsets
        if data_split is None:
            data_split = [0.7, 0.15, 0.15]
        random.shuffle()


    def __getitem__(self, idx):
        pass


    def get_dataset(self, dataset_path):

        # TODO: Other way to check if dataset is ok?

        if os.path.isdir(dataset_path):
            # If exists, check if contains files
            if os.listdir(dataset_path):
                # If contains files, check if files are audio.wav (dataset files)
                wav_file_list = glob.glob(os.path.join(dataset_path, '**', 'audio.wav'))
                if len(wav_file_list) == 0:
                    print(f"DATASET_ERROR: Found files in folder ({dataset_path}) but none pertaining to dataset. "
                          f"Exiting")
                    exit()
                elif len(wav_file_list) < MINIMUM_DATASET_LENGTH and self.is_debug:
                    print(f"DATASET_ERROR: Found files in dataset ({dataset_path}), but only {len(wav_file_list)} "
                          f"(under minimum limit {MINIMUM_DATASET_LENGTH}). Exiting")
                    exit()
                else:
                    # Load dataset paths
                    for wav_file in wav_file_list:
                        self.dataset_item_paths.append(os.path.dirname(wav_file))
                    return

        # If didn't exit or find dataset, create dataset
        dataset_builder = AudioDatasetBuilder(SOURCES, NOISES, dataset_path, MAX_SOURCES, SPEECH_AS_NOISE, SAMPLE_COUNT,
                                              self.is_debug)


    def get_subsets(self, data_split=None):
        if data_split is None:
            data_split = [0.7, 0.15, 0.15]


