import torch


class AudioDataset(torch.utils.data.Dataset):
    """
    Class handling usage of the audio dataset
    """

    def __init__(
            self,
            ROOT_PATH,
            sub_dataset_dir
    ):

        BASE_PATH = os.path.join(
            ROOT_PATH, Dataset.DATASET_DIR, sub_dataset_dir
        )
        self.IMAGES_DIR_PATH = os.path.join(BASE_PATH, Dataset.IMAGES_DIR)
        self.LABELS_DIR_PATH = os.path.join(BASE_PATH, Dataset.LABELS_DIR)

        self.images_paths = Dataset.get_multiple_workers_safe_list_of_paths(
            self.IMAGES_DIR_PATH
        )
