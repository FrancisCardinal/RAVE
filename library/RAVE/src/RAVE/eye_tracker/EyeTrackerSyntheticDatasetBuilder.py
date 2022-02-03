import os
from pathlib import Path
from shutil import copyfile
import random
from threading import Thread

from tqdm import tqdm

import cv2
from torchvision import transforms

from .EyeTrackerDatasetBuilder import EyeTrackerDatasetBuilder

from .EyeTrackerDataset import EyeTrackerDataset

from ..common.image_utils import tensor_to_opencv_image

(
    DATASET_DIR,
    IMAGES_DIR,
    LABELS_DIR,
    TRAINING_DIR,
    VALIDATION_DIR,
    TEST_DIR,
    IMAGES_FILE_EXTENSION,
) = (
    EyeTrackerDataset.DATASET_DIR,
    EyeTrackerDataset.IMAGES_DIR,
    EyeTrackerDataset.LABELS_DIR,
    EyeTrackerDataset.TRAINING_DIR,
    EyeTrackerDataset.VALIDATION_DIR,
    EyeTrackerDataset.TEST_DIR,
    EyeTrackerDataset.IMAGES_FILE_EXTENSION,
)


class EyeTrackerSyntheticDatasetBuilder(EyeTrackerDatasetBuilder):
    """
    This class builds the sub-datasets. It takes videos, extracts the frames
    and saves them on the disk, with the corresponding labels. It also
    applies data augmentation transforms to the training sub-dataset.
    """

    @staticmethod
    def create_images_datasets_with_synthetic_images():
        """
        Main method of the EyeTrackerSyntheticDatasetBuilder class.
        This method checks if the dataset as already been built, and builds it
        otherwise.
        """
        BUILDERS = EyeTrackerSyntheticDatasetBuilder.get_builders()
        if BUILDERS == -1:
            return

        print("dataset has NOT been found on disk, creating dataset")
        threads = []
        for builder in BUILDERS:
            thread = Thread(target=builder.generate_dataset)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    @staticmethod
    def get_builders():
        """
        Used to get the 3 EyeTrackerSyntheticDatasetBuilder objects
        (one for each sub-dataset)

        Returns:
            List of EyeTrackerSyntheticDatasetBuilder :
                The 3 EyeTrackerSyntheticDatasetBuilder objects
                (one for each sub-dataset)
        """
        SOURCE_DIR = os.path.join(
            EyeTrackerDataset.EYE_TRACKER_DIR_PATH, "synthetic_dataset"
        )

        TRAINING_PATH = os.path.join(
            EyeTrackerSyntheticDatasetBuilder.ROOT_PATH,
            EyeTrackerDataset.EYE_TRACKER_DIR_PATH,
            DATASET_DIR,
            TRAINING_DIR,
        )
        VALIDATION_PATH = os.path.join(
            EyeTrackerSyntheticDatasetBuilder.ROOT_PATH,
            EyeTrackerDataset.EYE_TRACKER_DIR_PATH,
            DATASET_DIR,
            VALIDATION_DIR,
        )
        TEST_PATH = os.path.join(
            EyeTrackerSyntheticDatasetBuilder.ROOT_PATH,
            EyeTrackerDataset.EYE_TRACKER_DIR_PATH,
            DATASET_DIR,
            TEST_DIR,
        )

        if os.path.isdir(TEST_PATH):
            print("dataset found on disk")
            return -1

        images_files = os.listdir(os.path.join(
            EyeTrackerDatasetBuilder.ROOT_PATH, SOURCE_DIR, IMAGES_DIR))
        random.Random(42).shuffle(images_files)

        train_size, val_size = 0.75, 0.15
        train_index_end = int(len(images_files)*train_size)
        val_index_end = train_index_end + int(len(images_files)*val_size)

        train_files = images_files[:train_index_end]
        val_files = images_files[train_index_end:val_index_end]
        test_files = images_files[val_index_end:]

        BUILDERS = [
            EyeTrackerSyntheticDatasetBuilderOfflineDataAugmentation(
                train_files,
                TRAINING_PATH,
                "training   dataset",
                EyeTrackerDataset.IMAGE_DIMENSIONS[1:3],
                SOURCE_DIR,
            ),
            EyeTrackerSyntheticDatasetBuilder(
                val_files,
                VALIDATION_PATH,
                "validation dataset",
                EyeTrackerDataset.IMAGE_DIMENSIONS[1:3],
                SOURCE_DIR,
            ),
            EyeTrackerSyntheticDatasetBuilder(
                test_files,
                TEST_PATH,
                "test       dataset",
                EyeTrackerDataset.IMAGE_DIMENSIONS[1:3],
                SOURCE_DIR,
            ),
        ]
        return BUILDERS

    def __init__(self, files, OUTPUT_DIR_PATH, log_name, IMAGE_DIMENSIONS, SOURCE_DIR):
        super().__init__([], OUTPUT_DIR_PATH, log_name, IMAGE_DIMENSIONS, SOURCE_DIR)
        self.INPUT_IMAGES_PATH = os.path.join(SOURCE_DIR, IMAGES_DIR)
        self.INPUT_LABELS_PATH = os.path.join(SOURCE_DIR, LABELS_DIR)

        self.files = files

    def generate_dataset(self):
        for file in tqdm(
            self.files, leave=False, desc=self.log_name
        ):
            filename = Path(file).stem
            frame = cv2.imread(os.path.join(
                self.INPUT_IMAGES_PATH, file))
            processed_frame = self.process_frame(frame)

            video_output_file_path = os.path.join(
                self.OUTPUT_IMAGES_PATH,
                filename + "." + IMAGES_FILE_EXTENSION,
            )
            output_frame = tensor_to_opencv_image(processed_frame)

            cv2.imwrite(video_output_file_path, output_frame)

            copyfile(os.path.join(
                self.INPUT_LABELS_PATH, filename + ".bin"),
                os.path.join(
                self.OUTPUT_LABELS_PATH, filename + ".bin")
            )


class EyeTrackerSyntheticDatasetBuilderOfflineDataAugmentation(
    EyeTrackerSyntheticDatasetBuilder
):
    """
    This class inherits from EyeTrackerSyntheticDatasetBuilder.
    It overwrites certain methods in order to do offline data augmentation.
    """

    def __init__(self, files, OUTPUT_DIR_PATH, log_name, IMAGE_DIMENSIONS, SOURCE_DIR):
        """
        Constructor of the EyeTrackerSyntheticDatasetBuilderOfflineDataAugmentation.
        Calls the parent constructor and defines the offline training transforms

        Args:
            VIDEOS (List of strings):
                The names of the videos that belong to this sub-dataset
            OUTPUT_DIR_PATH (String):
                Path of the directory that will contain the images and
                labels pairs
            log_name (String):
                Name to be displayed alongside the progress bar in the terminal
        """
        super().__init__(files, OUTPUT_DIR_PATH, log_name, IMAGE_DIMENSIONS, SOURCE_DIR)

        self.TRAINING_TRANSFORM = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15
                ),  # random
                transforms.GaussianBlur(3),  # random
                #transforms.RandomInvert(0.25),  # random
            ]
        )

    def process_frame(self, frame):
        """Calls the parent method, then applies some data augmentation operations.
           Used to perform offline data augmentation.
        Args:
            frame (numpy array): The frame that needs to be processed
        Returns:
            pytorch tensor: The processed frame
        """
        output_image_tensor = super().process_frame(frame)
        output_image_tensor = self.TRAINING_TRANSFORM(output_image_tensor)
        return output_image_tensor
