import os
import shutil
from threading import Thread
import random
from pathlib import Path
from shutil import copyfile
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
    CROP_SIZE = 150, 0, 450, 600

    @staticmethod
    def create_images_datasets_with_videos():
        """
        Main method of the EyeTrackerDatasetBuilder class.
        This method checks if the dataset as already been built, and builds it
        otherwise.
        """
        BUILDERS = EyeTrackerDatasetBuilder.get_builders()
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
    def get_builders():
        """
        Used to get the 3 EyeTrackerDatasetBuilder objects
        (one for each sub-dataset)

        Returns:
            List of EyeTrackerDatasetBuilder :
                The 3 EyeTrackerDatasetBuilder objects
                (one for each sub-dataset)
        """

        TRAINING_PATH = os.path.join(
            EyeTrackerDatasetBuilder.ROOT_PATH,
            EyeTrackerDataset.EYE_TRACKER_DIR_PATH,
            DATASET_DIR,
            TRAINING_DIR,
        )
        VALIDATION_PATH = os.path.join(
            EyeTrackerDatasetBuilder.ROOT_PATH,
            EyeTrackerDataset.EYE_TRACKER_DIR_PATH,
            DATASET_DIR,
            VALIDATION_DIR,
        )
        TEST_PATH = os.path.join(
            EyeTrackerDatasetBuilder.ROOT_PATH,
            EyeTrackerDataset.EYE_TRACKER_DIR_PATH,
            DATASET_DIR,
            TEST_DIR,
        )

        if os.path.isdir(TEST_PATH):
            print("dataset found on disk")
            return -1

        print("dataset has NOT been found on disk, creating dataset")

        VIDEO_UNPACKER = VideosUnpacker.get_builders()
        VIDEO_UNPACKER.create_images_of_one_video_group()

        SOURCE_DIR = VideosUnpacker.TMP_PATH

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
            EyeTrackerDatasetBuilder(
                test_files,
                TEST_PATH,
                "test      dataset",
                EyeTrackerDataset.IMAGE_DIMENSIONS[1:3],
                SOURCE_DIR,
            ),
        ]
        return BUILDERS

    def __init__(self, files, OUTPUT_DIR_PATH, log_name, IMAGE_DIMENSIONS, SOURCE_DIR, CROP_SIZE=None):
        super().__init__([], OUTPUT_DIR_PATH, log_name,
                         IMAGE_DIMENSIONS, SOURCE_DIR, CROP_SIZE)
        self.INPUT_IMAGES_PATH = os.path.join(SOURCE_DIR, IMAGES_DIR)
        self.INPUT_LABELS_PATH = os.path.join(SOURCE_DIR, LABELS_DIR)

        self.files = files

    def generate_dataset(self):
        self.video_frame_id = 0
        for file in tqdm(self.files, leave=False, desc=self.log_name):
            filename = Path(file).stem

            annotation = pickle.load(
                open(os.path.join(self.INPUT_LABELS_PATH, filename + ".bin"), "rb"))
            self.current_ellipse = NormalizedEllipse.get_from_list(annotation)

            frame = cv2.imread(os.path.join(self.INPUT_IMAGES_PATH, file))
            processed_frame = self.process_frame(frame)

            self.save_image_label_pair(
                filename, processed_frame, self.current_ellipse.to_list())
            self.video_frame_id += 1


class VideosUnpacker(DatasetBuilder):
    TMP_PATH = os.path.join(
        EyeTrackerDatasetBuilder.ROOT_PATH,
        EyeTrackerDataset.EYE_TRACKER_DIR_PATH,
        DATASET_DIR,
        'tmp',
    )

    @staticmethod
    def get_builders():
        VIDEOS = [
            "Amelie_1.avi",
            "Anthony_1.avi",
            "Felix_1.avi",
            "Francis_1.avi",
            "Olivier_1.avi",
            "Vincent_1.avi",
            "Jacob_1.avi",
            "Julien_1.avi",
            "Etienne_1.avi",
        ]

        BUILDER = VideosUnpacker(
            VIDEOS,
            VideosUnpacker.TMP_PATH,
            "Unpacking videos",
            EyeTrackerDataset.IMAGE_DIMENSIONS[1:3],
            os.path.join(EyeTrackerDataset.EYE_TRACKER_DIR_PATH,
                         "real_dataset"),
            EyeTrackerDatasetBuilder.CROP_SIZE,
        )

        return BUILDER

    def process_image_label_pair(
        self,
        processed_frame,
        file_name,
        ORIGINAL_HEIGHT,
        ORIGINAL_WIDTH
    ):
        """
        To process the image and label from LPW
        """
        self.current_ellipse.crop(
            ORIGINAL_HEIGHT, ORIGINAL_WIDTH, EyeTrackerDatasetBuilder.CROP_SIZE)
        self.save_image_label_pair(
            file_name, processed_frame, self.current_ellipse.to_list()
        )

    def parse_current_annotation(self, annotations):
        """
        Parses the current annotation to extract the parameters of
        the ellipse as defined by opencv

        Args:
            annotations (List of strings): All the annotations
            INPUT_IMAGE_WIDTH (int) The width of the input image
            INPUT_IMAGE_HEIGHT (int) The height of the input image

        Returns:
            bool: True if the parsing was a success, false if it wasn't.
        """
        annotation = annotations[str(self.annotation_line_index)]
        if (annotation[0] == -1) or (annotation[0] == -2):
            # The annotation files use '-1' and '-2' when the pupil is not
            # visible on a frame.
            return False

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
        self, VIDEOS, OUTPUT_DIR_PATH, log_name, IMAGE_DIMENSIONS, SOURCE_DIR, CROP_SIZE=None
    ):
        """
        Constructor of the TrainingDatasetBuilder.Calls the parent
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
            VIDEOS, OUTPUT_DIR_PATH, log_name, IMAGE_DIMENSIONS, SOURCE_DIR, CROP_SIZE
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
        """Calls the parent method, then applies some data augmentation operations.
           Used to perform offline data augmentation.
        Args:
            frame (numpy array): The frame that needs to be processed
        Returns:
            pytorch tensor: The processed frame
        """
        output_image_tensor = super().process_frame(frame)

        output_image_tensor = self.TRAINING_TRANSFORM(output_image_tensor)
        output_image_tensor = self.apply_translation_and_rotation(
            output_image_tensor)

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

        self.current_ellipse.rotate_around_image_center(phi)
        self.current_ellipse.h += x_offset
        self.current_ellipse.k += y_offset

        return output_image_tensor
