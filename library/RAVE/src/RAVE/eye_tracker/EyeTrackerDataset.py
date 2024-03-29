import os
import random

import torch
import torchvision
from torchvision import transforms

from ..common.image_utils import apply_image_translation
from ..common.image_utils import apply_image_rotation
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
    CROP_SIZE = 50, 0, 390, 520

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
        self.random = random.Random(42)

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
        Used to get an image and label pair (and domain).

        Args:
            idx (int): Index of the pair to get

        Returns:
            tuple: Image and label pair, and the domain of the image (is this
            image synthetic or real ?)
        """
        image_path = self.images_paths[idx]

        image, label = self.get_image_and_label_from_image_path(image_path)

        image = self.PRE_PROCESS_TRANSFORM(image)

        image = self.NORMALIZE_TRANSFORM(image)

        pupil_is_visible = label is not None

        if pupil_is_visible:
            label = torch.tensor(label)
        else:
            label = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])

        return image, label, pupil_is_visible

    @staticmethod
    def get_training_sub_dataset():
        """
        Used to get the training sub dataset

        Returns:
            Dataset: The training sub dataset
        """
        return EyeTrackerDatasetOnlineDataAugmentation(EyeTrackerDataset.TRAINING_DIR)

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
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # random
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
        image_path = self.images_paths[idx]

        image, label = self.get_image_and_label_from_image_path(image_path)

        image = self.PRE_PROCESS_TRANSFORM(image)

        output_image_tensor, phi = apply_image_rotation(image)

        output_image_tensor, x_offset, y_offset = apply_image_translation(output_image_tensor)
        pupil_is_visible = label is not None
        if pupil_is_visible:
            current_ellipse = NormalizedEllipse.get_from_list(label)
            current_ellipse.rotate_around_image_center(phi)
            current_ellipse.h += x_offset
            current_ellipse.k += y_offset
            label = current_ellipse.to_list()
            label = torch.tensor(label)

        else:
            label = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])

        image = self.NORMALIZE_TRANSFORM(output_image_tensor)
        pupil_is_visible = torch.tensor(pupil_is_visible).float()

        return image, label, pupil_is_visible


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

    def __init__(self, opencv_device, torch_device):
        super().__init__(None)
        self._video_feed = EyeTrackerVideoCapture(opencv_device)
        self._to_tensor_transform = transforms.ToTensor()
        self._resize_transform = transforms.Resize(self.IMAGE_DIMENSIONS[1:3])

        self._DEVICE = torch_device

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

        frame = self._to_tensor_transform(frame).to(self._DEVICE)

        top, left, height, width = EyeTrackerDataset.CROP_SIZE
        frame = torchvision.transforms.functional.crop(frame, top, left, height, width)

        frame = self._resize_transform(frame)
        frame = self.NORMALIZE_TRANSFORM(frame)

        return frame

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
        super().__init__(None)
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
