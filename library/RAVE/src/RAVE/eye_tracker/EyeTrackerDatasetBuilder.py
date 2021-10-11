import os
from threading import Thread

from torchvision import transforms

from ..common import DatasetBuilder
from .NormalizedEllipse import NormalizedEllipse
from .videos_and_dataset_association import (
    TRAINING_VIDEOS,
    VALIDATION_VIDEOS,
    TEST_VIDEOS,
)

from .EyeTrackerDataset import EyeTrackerDataset, IMAGE_DIMENSIONS

from ..common.image_utils import apply_image_translation, apply_image_rotation


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
    """This class builds the sub-datasets. It takes videos, extracts the frames
       and saves them on the disk, with the corresponding labels. It also
       applies data augmentation transforms to the training sub-dataset.
    """

    @staticmethod
    def create_images_datasets_with_LPW_videos():
        """Static methods ; Main method of the EyeTrackerDatasetBuilder class.
        This method checks if the dataset as already been built, and builds it
        otherwise.
        """
        BUILDERS = EyeTrackerDatasetBuilder.get_builders()
        if BUILDERS == -1:
            return

        print("dataset has NOT been found on disk, creating dataset")
        threads = []
        for builder in BUILDERS:
            thread = Thread(target=builder.create_images_of_one_video_group)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    @staticmethod
    def get_builders():
        """Static methods ; Used to get the 3 EyeTrackerDatasetBuilder objects
        (one for each sub-dataset)

        Returns:
            List of EyeTrackerDatasetBuilder:
                The 3 EyeTrackerDatasetBuilder objects
                (one for each sub-dataset)
        """
        SOURCE_DIR = os.path.join(
            EyeTrackerDataset.EYE_TRACKER_DIR_PATH, "LPW"
        )

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

        BUILDERS = [
            EyeTrackerDatasetBuilderOfflineDataAugmentation(
                TRAINING_VIDEOS,
                TRAINING_PATH,
                "training   dataset",
                IMAGE_DIMENSIONS[1:3],
                SOURCE_DIR,
            ),
            EyeTrackerDatasetBuilder(
                VALIDATION_VIDEOS,
                VALIDATION_PATH,
                "validation dataset",
                IMAGE_DIMENSIONS[1:3],
                SOURCE_DIR,
            ),
            EyeTrackerDatasetBuilder(
                TEST_VIDEOS,
                TEST_PATH,
                "test       dataset",
                IMAGE_DIMENSIONS[1:3],
                SOURCE_DIR,
            ),
        ]
        return BUILDERS

    def process_image_label_pair(
        self,
        processed_frame,
        file_name,
        annotations,
        INPUT_IMAGE_WIDTH,
        INPUT_IMAGE_HEIGHT,
    ):
        (
            angle,
            center_x,
            center_y,
            ellipse_width,
            ellipse_height,
        ) = self.parse_current_annotation(annotations)

        if angle == -1:
            # The annotation files use '-1' when the pupil is not visible
            # on a frame.
            return

        self.current_ellipse = NormalizedEllipse.get_normalized_ellipse_from_opencv_ellipse(
            center_x,
            ellipse_width,
            center_y,
            ellipse_height,
            angle,
            INPUT_IMAGE_WIDTH,
            INPUT_IMAGE_HEIGHT,
        )

        self.save_image_label_pair(
            file_name, processed_frame, self.current_ellipse.to_list()
        )

    def parse_current_annotation(self, annotations):
        """Parses the current annotation to extract the parameters of
           the ellipse as defined by opencv

        Args:
            annotations (List of strings): All the annotations

        Returns:
            quintuplet: The parameters of the ellipse as defined by opencv
        """
        annotation = annotations[self.annotation_line_index].split(";")
        # The end of the line has a ';' that must be removed
        annotation = annotation[0:-1]
        # To go from strings to floats
        annotation = [float(i) for i in annotation]
        (
            annotation_frame_id,
            angle,
            center_x,
            center_y,
            ellipse_width,
            ellipse_height,
        ) = annotation
        # To make sure that the current annotation really belongs to
        # the current frame
        assert self.video_frame_id == annotation_frame_id

        return angle, center_x, center_y, ellipse_width, ellipse_height


class EyeTrackerDatasetBuilderOfflineDataAugmentation(
    EyeTrackerDatasetBuilder
):
    """This class inherits from EyeTrackerDatasetBuilder.
    It overwrites certain methods in order to do offline data augmentation.
    """

    def __init__(
        self, VIDEOS, OUTPUT_DIR_PATH, log_name, IMAGE_DIMENSIONS, SOURCE_DIR
    ):
        """Constructor of the TrainingDatasetBuilder.Calls the parent
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
            VIDEOS, OUTPUT_DIR_PATH, log_name, IMAGE_DIMENSIONS, SOURCE_DIR
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

    def apply_translation_and_rotation(self, output_image_tensor):
        """A data augmentation operation, translates and rotates the frame
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
