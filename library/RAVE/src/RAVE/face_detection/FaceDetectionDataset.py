import os
from ..common.Dataset import Dataset


class FaceDetectionDataset(Dataset):
    """
    Class that handles pairs of images and labels that are on disk

    Args:
        sub_dataset_dir (String): Name of the directory of the sub-dataset
    """

    FACE_DETECTION_DIR_PATH = os.path.join("RAVE", "face_detection")
    TRAINING_MEAN, TRAINING_STD = [-0.0648, -0.0950, 0.0119], [
        1.0695,
        1.0609,
        1.0581,
    ]
    IMAGE_DIMENSIONS = (3, 800, 800)

    def __init__(self, sub_dataset_dir):
        super().__init__(
            FaceDetectionDataset.TRAINING_MEAN,
            FaceDetectionDataset.TRAINING_STD,
            FaceDetectionDataset.FACE_DETECTION_DIR_PATH,
            sub_dataset_dir,
            FaceDetectionDataset.IMAGE_DIMENSIONS,
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
        img, label = super().__getitem__(idx)

        return img, label, label.shape

    @staticmethod
    def get_training_sub_dataset():
        """
        Used to get the training sub dataset

        Returns:
            Dataset: The training sub dataset
        """
        return FaceDetectionDataset(FaceDetectionDataset.TRAINING_DIR)

    @staticmethod
    def get_validation_sub_dataset():
        """
        Used to get the validation sub dataset

        Returns:
            Dataset: The validation sub dataset
        """
        return FaceDetectionDataset(FaceDetectionDataset.VALIDATION_DIR)

    @staticmethod
    def get_test_sub_dataset():
        """
        Used to get the test sub dataset

        Returns:
            Dataset: The test sub dataset
        """
        return FaceDetectionDataset(FaceDetectionDataset.TEST_DIR)
