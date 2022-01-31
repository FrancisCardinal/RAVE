import os

import torch
from torchvision import transforms

from ..common.image_utils import apply_image_translation, apply_image_rotation
from ..common.Dataset import Dataset
from .NormalizedEllipse import NormalizedEllipse


class EyeTrackerDataset(Dataset):
    """
    Class that handles pairs of images and labels that are on disk

    Args:
        sub_dataset_dir (String): Name of the directory of the sub-dataset
    """

    EYE_TRACKER_DIR_PATH = os.path.join("RAVE", "eye_tracker")
    TRAINING_MEAN, TRAINING_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    IMAGE_DIMENSIONS = (1, 240, 320)

    def __init__(self, sub_dataset_dir):
        super().__init__(
            EyeTrackerDataset.TRAINING_MEAN,
            EyeTrackerDataset.TRAINING_STD,
            EyeTrackerDataset.EYE_TRACKER_DIR_PATH,
            sub_dataset_dir,
            EyeTrackerDataset.IMAGE_DIMENSIONS,
        )

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
        label = label["ellipse"]

        image = self.PRE_PROCESS_TRANSFORM(image)

        image = self.NORMALIZE_TRANSFORM(image)
        label = torch.tensor(label)

        return image, label

    @staticmethod
    def get_training_sub_dataset():
        """
        Used to get the training sub dataset

        Returns:
            Dataset: The training sub dataset
        """
        return EyeTrackerDataset(EyeTrackerDataset.TRAINING_DIR)

    @staticmethod
    def get_validation_sub_dataset():
        """
        Used to get the validation sub dataset

        Returns:
            Dataset: The validation sub dataset
        """
        return EyeTrackerDataset(EyeTrackerDataset.VALIDATION_DIR)

    @staticmethod
    def get_test_sub_dataset():
        """
        Used to get the test sub dataset

        Returns:
            Dataset: The test sub dataset
        """
        return EyeTrackerDataset(EyeTrackerDataset.TEST_DIR)


class EyeTrackerDatasetOnlineDataAugmentation(Dataset):
    """
    This class inherits from Dataset. It overwrites certain methods in
    order to do online data augmentation.

    Args:
        sub_dataset_dir (String): Name of the directory of the sub-dataset
    """

    def __init__(self, sub_dataset_dir):
        super().__init__(sub_dataset_dir)

        self.TRAINING_TRANSFORM = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                ),  # random
                transforms.GaussianBlur(3),  # random
                transforms.RandomInvert(0.25),  # random
            ]
        )

    def __getitem__(self, idx):
        """
        Method of the Dataset class that must be overwritten by this class.
        Used to get an image and label pair. Before returning the image and
        label pair, this class performs online data augmentation.

        Args:
            idx (int): Index of the pair to get

        Returns:
            tuple: Image and label pair
        """
        image, label = self.get_image_and_label_on_disk(idx)
        label = label["ellipse"]

        image = self.PRE_PROCESS_TRANSFORM(image)

        output_image_tensor = self.TRAINING_TRANSFORM(image)

        output_image_tensor, phi = apply_image_rotation(output_image_tensor)

        output_image_tensor, x_offset, y_offset = apply_image_translation(
            output_image_tensor
        )

        current_ellipse = NormalizedEllipse.get_from_list(label)
        current_ellipse.rotate_around_image_center(phi)
        current_ellipse.h += x_offset
        current_ellipse.k += y_offset
        label = current_ellipse.to_list()

        image = Dataset.NORMALIZE_TRANSFORM(output_image_tensor)
        label = torch.tensor(label)

        return image, label
