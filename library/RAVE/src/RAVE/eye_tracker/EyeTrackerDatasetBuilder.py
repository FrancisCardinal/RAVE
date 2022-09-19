import os
from os.path import isfile, join
import shutil
from threading import Thread
import random
from pathlib import Path
from tqdm import tqdm

import cv2
import pickle
from torchvision import transforms

from ..common import DatasetBuilder
from ..common.image_utils import apply_image_translation, apply_image_rotation

from .NormalizedEllipse import NormalizedEllipse

from .EyeTrackerDataset import EyeTrackerDataset

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


class EyeTrackerDatasetBuilder(DatasetBuilder):
    """
    This class builds the sub-datasets. It takes videos, extracts the frames
    and saves them on the disk, with the corresponding labels. It also
    applies data augmentation transforms to the training sub-dataset.
    """

    @staticmethod
    def create_datasets(VIDEOS_DIR, IS_SECONDARY_DATASET=False):
        """
        Main method of the EyeTrackerDatasetBuilder class.
        This method checks if the dataset as already been built, and builds it
        otherwise.
        """
        BUILDERS = EyeTrackerDatasetBuilder.get_builders(
            VIDEOS_DIR, IS_SECONDARY_DATASET
        )
        if BUILDERS == -1:
            return False

        threads = []
        for builder in BUILDERS:
            thread = Thread(target=builder.generate_dataset)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        shutil.rmtree(VideosUnpacker.TMP_PATH)

        return True

    @staticmethod
    def get_builders(VIDEOS_DIR, IS_SECONDARY_DATASET):
        """
        Used to get the 3 EyeTrackerDatasetBuilder objects
        (one for each sub-dataset)

        Returns:
            List of EyeTrackerDatasetBuilder :
                The 3 EyeTrackerDatasetBuilder objects
                (one for each sub-dataset)
        """

        TRAINING_PATH = join(
            EyeTrackerDatasetBuilder.ROOT_PATH,
            EyeTrackerDataset.EYE_TRACKER_DIR_PATH,
            DATASET_DIR,
            TRAINING_DIR,
        )
        VALIDATION_PATH = join(
            EyeTrackerDatasetBuilder.ROOT_PATH,
            EyeTrackerDataset.EYE_TRACKER_DIR_PATH,
            DATASET_DIR,
            VALIDATION_DIR,
        )
        TEST_PATH = join(
            EyeTrackerDatasetBuilder.ROOT_PATH,
            EyeTrackerDataset.EYE_TRACKER_DIR_PATH,
            DATASET_DIR,
            TEST_DIR,
        )

        if os.path.isdir(TEST_PATH) and not IS_SECONDARY_DATASET:
            print("dataset found on disk")
            return -1

        print("dataset has NOT been found on disk, creating dataset")

        CROP_SIZE = EyeTrackerDataset.CROP_SIZE
        if IS_SECONDARY_DATASET:
            CROP_SIZE = 150, 0, 450, 600

        VIDEO_UNPACKER = VideosUnpacker.get_builders(VIDEOS_DIR, CROP_SIZE)
        VIDEO_UNPACKER.create_images_of_one_video_group()

        SOURCE_DIR = VideosUnpacker.TMP_PATH

        images_files = os.listdir(
            join(EyeTrackerDatasetBuilder.ROOT_PATH, SOURCE_DIR, IMAGES_DIR)
        )
        random.Random(42).shuffle(images_files)

        train_size, val_size = 0.75, 0.15

        if IS_SECONDARY_DATASET:
            train_size += 0.10

        train_index_end = int(len(images_files) * train_size)
        val_index_end = train_index_end + int(len(images_files) * val_size)

        train_files = images_files[:train_index_end]
        val_files = images_files[train_index_end:val_index_end]

        BUILDERS = [
            EyeTrackerDatasetBuilder(
                train_files,
                TRAINING_PATH,
                "training   dataset",
                EyeTrackerDataset.IMAGE_DIMENSIONS[1:3],
                SOURCE_DIR,
            ),
            EyeTrackerDatasetBuilder(
                val_files,
                VALIDATION_PATH,
                "validation dataset",
                EyeTrackerDataset.IMAGE_DIMENSIONS[1:3],
                SOURCE_DIR,
            ),
        ]

        if not IS_SECONDARY_DATASET:
            test_files = images_files[val_index_end:]
            BUILDERS.append(
                EyeTrackerDatasetBuilder(
                    test_files,
                    TEST_PATH,
                    "test      dataset",
                    EyeTrackerDataset.IMAGE_DIMENSIONS[1:3],
                    SOURCE_DIR,
                )
            )

        return BUILDERS

    def __init__(
        self,
        files,
        OUTPUT_DIR_PATH,
        log_name,
        IMAGE_DIMENSIONS,
        SOURCE_DIR,
        CROP_SIZE=None,
    ):
        """Constructor of the EyeTrackerDatasetBuilder class

        Args:
            files (List): Filenames of the elements of the dataset
            OUTPUT_DIR_PATH (string): Directory where we should put the dataset
            log_name (string): Name that sould be displayed in the logs to
               represent this dataset while its being builded
            IMAGE_DIMENSIONS (tuple): (Height, Width) Output image dimension
            SOURCE_DIR (string): Directory where we can get the video
            CROP_SIZE (tuple, optional): 4 values that correspond to a crop
               operation : top (y coordinate), left (x coordinate),
               height, width. Defaults to None. If None, do not perform a crop.
        """
        super().__init__(
            [],
            OUTPUT_DIR_PATH,
            log_name,
            IMAGE_DIMENSIONS,
            SOURCE_DIR,
            CROP_SIZE,
        )
        self.INPUT_IMAGES_PATH = join(SOURCE_DIR, IMAGES_DIR)
        self.INPUT_LABELS_PATH = join(SOURCE_DIR, LABELS_DIR)

        self.files = files

    def generate_dataset(self):
        """Generates the dataset"""
        self.video_frame_id = 0
        for file in tqdm(self.files, leave=False, desc=self.log_name):
            filename = Path(file).stem

            annotation = pickle.load(
                open(
                    join(self.INPUT_LABELS_PATH, filename + ".bin"),
                    "rb",
                )
            )
            if annotation is not None:
                self.current_ellipse = NormalizedEllipse.get_from_list(
                    annotation
                )
            else:
                self.current_ellipse = None

            frame = cv2.imread(join(self.INPUT_IMAGES_PATH, file))
            processed_frame = self.process_frame(frame)

            if self.current_ellipse is not None:
                label = self.current_ellipse.to_list()
            else:
                label = None

            self.save_image_label_pair(filename, processed_frame, label)
            self.video_frame_id += 1


class VideosUnpacker(DatasetBuilder):
    """Class that is used to unpack a video into its constituting frames
    and dumps them into a temporary directory so that it can be used
    by the EyeTrackerDatasetBuilder class
    """

    TMP_PATH = join(
        EyeTrackerDatasetBuilder.ROOT_PATH,
        EyeTrackerDataset.EYE_TRACKER_DIR_PATH,
        DATASET_DIR,
        "tmp",
    )

    @staticmethod
    def get_builders(VIDEOS_DIR, CROP_SIZE):
        """Get the builder

        Returns:
            VideosUnpacker: The VideosUnpacker object
        """
        VIDEOS_DIR_PATH = join(
            EyeTrackerDatasetBuilder.ROOT_PATH,
            EyeTrackerDataset.EYE_TRACKER_DIR_PATH,
            VIDEOS_DIR,
            "videos",
        )
        VIDEOS = [
            f
            for f in os.listdir(VIDEOS_DIR_PATH)
            if isfile(join(VIDEOS_DIR_PATH, f))
        ]

        BUILDER = VideosUnpacker(
            VIDEOS,
            VideosUnpacker.TMP_PATH,
            "Unpacking videos",
            EyeTrackerDataset.IMAGE_DIMENSIONS[1:3],
            join(EyeTrackerDataset.EYE_TRACKER_DIR_PATH, VIDEOS_DIR),
            CROP_SIZE,
        )

        return BUILDER

    def process_image_label_pair(
        self, processed_frame, file_name, ORIGINAL_HEIGHT, ORIGINAL_WIDTH
    ):
        """
        To process the image and label from the target dataset
        """
        if self.current_ellipse is not None:
            self.current_ellipse.crop(
                ORIGINAL_HEIGHT, ORIGINAL_WIDTH, EyeTrackerDataset.CROP_SIZE
            )
            label = self.current_ellipse.to_list()
        else:
            label = None

        self.save_image_label_pair(file_name, processed_frame, label)

    def parse_current_annotation(self, annotations):
        """
        Parses the current annotation to extract the parameters of
        the ellipse as defined by the annotation tool

        Args:
            annotations (List of strings): All the annotations
            INPUT_IMAGE_WIDTH (int) The width of the input image
            INPUT_IMAGE_HEIGHT (int) The height of the input image

        Returns:
            bool: True if the parsing was a success, false if it wasn't.
        """
        annotation = annotations[str(self.annotation_line_index)]
        if annotation[0] == -2:
            # '-2' means the frame was skipped (no annotation)
            return False

        if annotation[0] == -1:
            # -1 means no pupil is visible (the person is blinking / ...)
            self.current_ellipse = None
        else:
            self.current_ellipse = NormalizedEllipse.get_from_list(annotation)

        return True


class EyeTrackerDatasetBuilderOfflineDataAugmentation(
    EyeTrackerDatasetBuilder
):
    """
    This class inherits from EyeTrackerDatasetBuilder.
    It overwrites certain methods in order to do offline data augmentation.
    """

    def __init__(
        self,
        VIDEOS,
        OUTPUT_DIR_PATH,
        log_name,
        IMAGE_DIMENSIONS,
        SOURCE_DIR,
        CROP_SIZE=None,
    ):
        """
        Constructor of the TrainingDatasetBuilder. Calls the parent
        constructor and defines the training transforms

        Args:
            VIDEOS (List of strings):
                The names of the videos that belong to this sub-dataset
            OUTPUT_DIR_PATH (String):
                Path of the directory that will contain the images and
                labels pairs
            log_name (String):
                Name to be displayed alongside the progress bar in the terminal
        """
        super().__init__(
            VIDEOS,
            OUTPUT_DIR_PATH,
            log_name,
            IMAGE_DIMENSIONS,
            SOURCE_DIR,
            CROP_SIZE,
        )

        self.TRAINING_TRANSFORM = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                ),  # random
                transforms.GaussianBlur(3),  # random
                transforms.RandomInvert(0.25),  # random
            ]
        )

    def process_frame(self, frame):
        """Calls the parent method, then applies some data augmentation
           operations. Used to perform offline data augmentation.
        Args:
            frame (numpy array): The frame that needs to be processed
        Returns:
            pytorch tensor: The processed frame
        """
        output_image_tensor = super().process_frame(frame)

        output_image_tensor = self.TRAINING_TRANSFORM(output_image_tensor)
        output_image_tensor = self.apply_translation_and_rotation(
            output_image_tensor
        )

        return output_image_tensor

    def apply_translation_and_rotation(self, output_image_tensor):
        """
        A data augmentation operation, translates and rotates the frame
        randomly and reflects that change on the corresponding ellipse

        Args:
            output_image_tensor (pytorch tensor):
                The frame on which to apply the translation and rotation

        Returns:
            pytorch tensor:
                The translated and rotated frame
        """
        output_image_tensor, phi = apply_image_rotation(output_image_tensor)

        output_image_tensor, x_offset, y_offset = apply_image_translation(
            output_image_tensor
        )
        if self.current_ellipse is not None:
            self.current_ellipse.rotate_around_image_center(phi)
            self.current_ellipse.h += x_offset
            self.current_ellipse.k += y_offset

        return output_image_tensor
