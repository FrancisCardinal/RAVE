import os
import random

import torch
from torchvision import transforms

from PIL import Image
import numpy as np

from ..common.image_utils import apply_image_translation, apply_image_rotation
from ..common.Dataset import Dataset
from ..eye_tracker.EyeTrackerVideoCapture import EyeTrackerVideoCapture
from .NormalizedEllipse import NormalizedEllipse


class EyeTrackerDataset(Dataset):
    """
    Class that handles pairs of images and labels that are on disk

    Args:
        sub_dataset_dir (String): Name of the directory of the sub-dataset
    """

    EYE_TRACKER_DIR_PATH = os.path.join("RAVE", "eye_tracker")
    TRAINING_MEAN, TRAINING_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    IMAGE_DIMENSIONS = (3, 240, 320)
    CROP_SIZE = 150, 0, 450, 600
    SYNTHETIC_DOMAIN = 0
    REAL_DOMAIN = 1

    def __init__(self, sub_dataset_dir):
        """Constructor of the EyeTrackerDataset class

        Args:
            sub_dataset_dir (string): Name of the sub dataset directory
        """
        super().__init__(
            EyeTrackerDataset.TRAINING_MEAN,
            EyeTrackerDataset.TRAINING_STD,
            EyeTrackerDataset.EYE_TRACKER_DIR_PATH,
            sub_dataset_dir,
            EyeTrackerDataset.IMAGE_DIMENSIONS,
        )

        self.real_images_paths, self.synthetic_images_paths = [], []
        for image_path in self.images_paths:
            if "synthetic" in image_path:
                self.synthetic_images_paths.append(image_path)
            else:
                self.real_images_paths.append(image_path)

        self.nb_synthetic_images = len(self.synthetic_images_paths)
        self.nb_real_images = len(self.real_images_paths)

        self.real_images_paths = np.array(
            [str(i) for i in self.real_images_paths], dtype=np.str
        )
        self.synthetic_images_paths = np.array(
            [str(i) for i in self.synthetic_images_paths], dtype=np.str
        )

        self.random = random.Random(42)

    def __len__(self):
        """
        Method of the Dataset class that must be overwritten by this class.
        Used to get the number of elements in the dataset

        Returns:
            int: The number of elements in the dataset
        """
        return self.nb_synthetic_images + self.nb_real_images

    def __getitem__(self, idx):
        """
        Method of the Dataset class that must be overwritten by this class.
        Used to get an image and label pair (and domain).

        Args:
            idx (int): Index of the pair to get

        Returns:
            tuple: Image and label pair, and the domain of the image (is this
            image synthetic or real ?)
        """
        image_path, domain = self.torch_index_to_image_path_and_domain(idx)

        image, label = self.get_image_and_label_from_image_path(image_path)

        image = self.PRE_PROCESS_TRANSFORM(image)

        image = self.NORMALIZE_TRANSFORM(image)
        label = torch.tensor(label)
        domain = torch.tensor(domain).float()

        return image, label, domain

    def torch_index_to_image_path_and_domain(self, idx):
        """Determines if a given index corresponds to a synthetic or real image
           (i.e its domain) and returns the image's path and domain

        Args:
            idx (int): Index of the image

        Returns:
            tuple: The image's path and domain
        """
        image_path, domain = None, None
        if idx < self.nb_synthetic_images:
            # idx is one of our self.nb_synthetic_images real imag
            image_path = self.synthetic_images_paths[idx]
            domain = EyeTrackerDataset.SYNTHETIC_DOMAIN

        else:
            # idx is one of our self.nb_real_images real image
            image_path = self.real_images_paths[idx - self.nb_synthetic_images]
            domain = EyeTrackerDataset.REAL_DOMAIN

        return image_path, domain

    @staticmethod
    def get_training_sub_dataset():
        """
        Used to get the training sub dataset

        Returns:
            Dataset: The training sub dataset
        """
        return EyeTrackerDatasetOnlineDataAugmentation(
            EyeTrackerDataset.TRAINING_DIR
        )

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


class EyeTrackerDatasetOnlineDataAugmentation(EyeTrackerDataset):
    """
    This class inherits from Dataset. It overwrites certain methods in
    order to do online data augmentation.

    Args:
        sub_dataset_dir (String): Name of the directory of the sub-dataset
    """

    def __init__(self, sub_dataset_dir):
        """Constructor of the EyeTrackerDatasetOnlineDataAugmentation class

        Args:
            sub_dataset_dir (string): Name of the sub dataset directory
        """
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
        Used to get an image and label pair (and domain). Before returning the
        image and label pair, this class performs online data augmentation.

        Args:
            idx (int): Index of the pair to get

        Returns:
            tuple: Image and label pair, and the domain of the image (is this
            image synthetic or real ?)
        """
        image_path, domain = self.torch_index_to_image_path_and_domain(idx)

        image, label = self.get_image_and_label_from_image_path(image_path)

        image = self.PRE_PROCESS_TRANSFORM(image)

        output_image_tensor, phi = apply_image_rotation(image)

        output_image_tensor, x_offset, y_offset = apply_image_translation(
            output_image_tensor
        )

        current_ellipse = NormalizedEllipse.get_from_list(label)
        current_ellipse.rotate_around_image_center(phi)
        current_ellipse.h += x_offset
        current_ellipse.k += y_offset
        label = current_ellipse.to_list()

        image = self.NORMALIZE_TRANSFORM(output_image_tensor)
        label = torch.tensor(label)
        domain = torch.tensor(domain).float()

        return image, label, domain


class EyeTrackerInferenceDataset(EyeTrackerDataset):
    """
    Class that handles the management of the frames
    of a video on disk or of a real time video feed
    for inference.

    Args:
        opencv_device (String): OpenCV device, that is, a path
                         to a video or a opencv device
                         index
    """

    ACQUISITION_WIDTH, ACQUISITION_HEIGHT = (
        EyeTrackerVideoCapture.ACQUISITION_WIDTH,
        EyeTrackerVideoCapture.ACQUISITION_HEIGHT,
    )

    def __init__(self, opencv_device):
        super().__init__("test")  # TODO FC : Find a more elegant solution
        self._video_feed = EyeTrackerVideoCapture(opencv_device)

    def __len__(self):
        """
        Method of the Dataset class that must be overwritten by this class.
        Used to get the number of elements in the dataset

        Returns:
            int: The number of elements in the dataset (1, as we are only
            interested in the most recent frame (which changes over time))
        """
        return 1

    def __getitem__(self, idx):
        """
        Method of the Dataset class that must be overwritten by this class.
        Used to get an image

        Args:
            idx (int): Index of the pair to get, ignored

        Returns:
            torch.tensor: The most recent image
        """
        frame = self._video_feed.read()

        frame = Image.fromarray(frame, "RGB")
        image = self.PRE_PROCESS_TRANSFORM(frame)
        image = self.NORMALIZE_TRANSFORM(image)

        return image

    def end(self):
        """Should be called at the end of the program to free the opencv
        device"""
        self._video_feed.end()


class EyeTrackerFilm(EyeTrackerDataset):
    """
    Class that handles the management of the frames
    of a live video.

    Args:
        opencv_device (String): OpenCV device index
    """

    ACQUISITION_WIDTH, ACQUISITION_HEIGHT = (
        EyeTrackerVideoCapture.ACQUISITION_WIDTH,
        EyeTrackerVideoCapture.ACQUISITION_HEIGHT,
    )

    def __init__(self, opencv_device):
        super().__init__("test")  # TODO FC : Find a more elegant solution
        self._video_feed = EyeTrackerVideoCapture(opencv_device)

    def __len__(self):
        """
        Method of the Dataset class that must be overwritten by this class.
        Used to get the number of elements in the dataset

        Returns:
            int: The number of elements in the dataset (1, as we are only
            interested in the most recent frame (which changes over time))
        """
        return 1

    def __getitem__(self, idx):
        """
        Method of the Dataset class that must be overwritten by this class.
        Used to get an image

        Args:
            idx (int): Index of the pair to get, ignored

        Returns:
            torch.tensor: The most recent frame
        """
        frame = self._video_feed.read()
        return frame

    def end(self):
        """Should be called at the end of the program to free the opencv
        device"""
        self._video_feed.end()
