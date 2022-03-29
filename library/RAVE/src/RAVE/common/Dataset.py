from abc import abstractmethod
import os

import torch
from torchvision import transforms

import numpy as np
import random

from PIL import Image
import pickle


class Dataset(torch.utils.data.Dataset):
    """
    Class that handles pairs of images and labels that are on disk

    Args:
        TRAINING_MEAN (float): Training dataset mean pixel value.
        TRAINING_STD (float): Training dataset std pixel value.
        ROOTH_PATH (String): Root path of the dataset
        sub_dataset_dir (String): Name of the directory of the sub-dataset
        IMAGE_DIMENSIONS (Tuple):
            Image dimensions with shape (3, height, width)
    """

    DATASET_DIR = "dataset"

    IMAGES_DIR = "images"
    LABELS_DIR = "labels"

    TRAINING_DIR = "training"
    VALIDATION_DIR = "validation"
    TEST_DIR = "test"

    # TODO-jkealey: Dataset should not defined the file extension
    IMAGES_FILE_EXTENSION = "png"

    def __init__(
        self,
        TRAINING_MEAN,
        TRAINING_STD,
        ROOT_PATH,
        sub_dataset_dir,
        IMAGE_DIMENSIONS,
    ):
        self.TRAINING_MEAN, self.TRAINING_STD = TRAINING_MEAN, TRAINING_STD
        self.IMAGE_DIMENSIONS = IMAGE_DIMENSIONS
        self.PRE_PROCESS_TRANSFORM = transforms.Compose(
            [transforms.Resize(IMAGE_DIMENSIONS[1:3]), transforms.ToTensor(), ]
        )
        self.NORMALIZE_TRANSFORM = transforms.Normalize(
            mean=TRAINING_MEAN, std=TRAINING_STD
        )

        BASE_PATH = os.path.join(
            ROOT_PATH, Dataset.DATASET_DIR, sub_dataset_dir
        )
        self.IMAGES_DIR_PATH = os.path.join(BASE_PATH, Dataset.IMAGES_DIR)
        self.LABELS_DIR_PATH = os.path.join(BASE_PATH, Dataset.LABELS_DIR)

        self.images_paths = Dataset.get_multiple_workers_safe_list_of_paths(
            self.IMAGES_DIR_PATH
        )

    def __len__(self):
        """
        Method of the Dataset class that must be overwritten by this class.
        Used to get the number of elements in the dataset

        Returns:
            int: The number of elements in the dataset
        """
        return len(self.images_paths)

    def __getitem__(self, idx):
        """
        Method of the Dataset class that must be overwritten by this class.
        Used to get an image and label pair

        Args:
            idx (int): Index of the pair to get

        Returns:
            tuple: Image and label pair
        """
        image, label = self.get_image_and_label_on_disk(idx)

        image = self.PRE_PROCESS_TRANSFORM(image)

        image = self.NORMALIZE_TRANSFORM(image)
        label = torch.tensor(label)

        return image, label

    def get_image_and_label_on_disk(self, idx):
        """
        Gets an image and label pair on disk

        Args:
            idx (int): Index of the image and label pair

        Returns:
            tuple: Image and label pair
        """
        image_path = self.images_paths[idx]
        return self.get_image_and_label_from_image_path(image_path)

    def get_image_and_label_from_image_path(self, image_path):
        """
        Gets an image and label pair on disk

        Args:
            image_path (str): Path of the image

        Returns:
            tuple: Image and label pair
        """
        image_path = os.path.join(self.IMAGES_DIR_PATH, image_path)
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        image = Image.open(image_path)

        label_path = os.path.join(self.LABELS_DIR_PATH, file_name + ".bin")
        label = pickle.load(open(label_path, "rb"))

        return image, label

    @staticmethod
    def get_multiple_workers_safe_list_of_paths(directory):
        """
        Used to build a list of paths. This method prevents a memory
        leak that happens with list of strings and multiple dataloader
        workers
        https://gist.github.com/mprostock/2850f3cd465155689052f0fa3a177a50

        Args:
            directory (String):
                The directory of which to get the paths of its files

        Returns:
            List: The list of paths
        """
        paths = os.listdir(directory)
        # Just to make sure elements of a given batch don't look alike
        random.Random(42).shuffle(paths)
        return np.array([str(i) for i in paths], dtype=np.str)

    @staticmethod
    @abstractmethod
    def get_training_sub_dataset():
        """
        Used to get the training sub dataset

        Returns:
            Dataset: The training sub dataset
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_validation_sub_dataset():
        """
        Used to get the validation sub dataset

        Returns:
            Dataset: The validation sub dataset
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_test_sub_dataset():
        """
        Used to get the test sub dataset

        Returns:
            Dataset: The test sub dataset
        """
        raise NotImplementedError
