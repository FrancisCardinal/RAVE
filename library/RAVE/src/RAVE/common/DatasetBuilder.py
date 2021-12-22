import os
import cv2
from abc import ABC, abstractmethod
from PIL import Image
import pickle
from tqdm import tqdm

from torchvision import transforms

from .Dataset import Dataset

from .image_utils import tensor_to_opencv_image


(
    DATASET_DIR,
    IMAGES_DIR,
    LABELS_DIR,
    TRAINING_DIR,
    VALIDATION_DIR,
    TEST_DIR,
    IMAGES_FILE_EXTENSION,
) = (
    Dataset.DATASET_DIR,
    Dataset.IMAGES_DIR,
    Dataset.LABELS_DIR,
    Dataset.TRAINING_DIR,
    Dataset.VALIDATION_DIR,
    Dataset.TEST_DIR,
    Dataset.IMAGES_FILE_EXTENSION,
)


class DatasetBuilder(ABC):
    """
    This class builds the sub-datasets. It takes videos, extracts the frames
    and saves them on the disk, with the corresponding labels.
    It also applies data augmentation transforms to the training sub-dataset

    Args:
        VIDEOS (List of strings):
            The names of the videos that belong to this sub-dataset
        OUTPUT_DIR_PATH (String):
            Path of the directory that will contain the images and labels
            pairs
        log_name (String):
            Name to be displayed alongside the progress bar in the terminal
    """

    VIDEOS_DIR = "videos"
    ANNOTATIONS_DIR = "annotations"

    ROOT_PATH = os.getcwd()

    def __init__(
        self, VIDEOS, OUTPUT_DIR_PATH, log_name, IMAGE_DIMENSIONS, SOURCE_DIR
    ):
        self.VIDEOS = VIDEOS
        self.log_name = log_name

        self.VIDEOS_PATH = os.path.join(
            DatasetBuilder.ROOT_PATH, SOURCE_DIR, DatasetBuilder.VIDEOS_DIR
        )
        self.ANNOTATIONS_PATH = os.path.join(
            DatasetBuilder.ROOT_PATH,
            SOURCE_DIR,
            DatasetBuilder.ANNOTATIONS_DIR,
        )

        self.OUTPUT_IMAGES_PATH = os.path.join(OUTPUT_DIR_PATH, IMAGES_DIR)
        self.OUTPUT_LABELS_PATH = os.path.join(OUTPUT_DIR_PATH, LABELS_DIR)
        DatasetBuilder.create_directory_if_does_not_exist(OUTPUT_DIR_PATH)
        DatasetBuilder.create_directory_if_does_not_exist(
            self.OUTPUT_IMAGES_PATH
        )
        DatasetBuilder.create_directory_if_does_not_exist(
            self.OUTPUT_LABELS_PATH
        )

        self.RESIZE_TRANSFORM = transforms.Compose(
            [transforms.Resize(IMAGE_DIMENSIONS), transforms.ToTensor()]
        )

    @staticmethod
    @abstractmethod
    def get_builders():
        """
        Used to get the 3 DatasetBuilder objects (one for each sub-dataset)

        Returns:
            List of DatasetBuilder: The 3 DatasetBuilder objects (one for
            each sub-dataset)
        """
        raise NotImplementedError

    @staticmethod
    def create_directory_if_does_not_exist(path):
        """
        Creates a directory if it does not exist on the disk

        Args:
            path (string): The path of the directory to create
        """
        if not os.path.isdir(path):
            os.makedirs(path)

    def create_images_of_one_video_group(self):
        """
        Gets the info of one video, then creates the images and labels pair
        of the video and save them to disk
        """
        for video_file_name in tqdm(
            self.VIDEOS, leave=False, desc=self.log_name
        ):
            video_path = os.path.join(self.VIDEOS_PATH, video_file_name)

            if not os.path.isfile(video_path):
                return

            file_name = os.path.splitext(os.path.basename(video_file_name))[0]
            annotations_file = open(
                os.path.join(self.ANNOTATIONS_PATH, file_name + ".txt"), "r"
            )
            annotations = annotations_file.readlines()

            self.create_images_dataset_with_one_video(
                file_name, video_path, annotations
            )

    def create_images_dataset_with_one_video(
        self, file_name, video_path, annotations
    ):
        """
        Creates the images and labels pair of the video and saves
        them to disk

        Args:
            file_name (String): Name of the video
            video_path (String): Path to the video
            annotations (List of strings):
                One string per frame. Each string represents the ellipse that
                is present on the frame
        """
        cap = cv2.VideoCapture(video_path)

        INPUT_IMAGE_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        INPUT_IMAGE_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.annotation_line_index = 0
        self.video_frame_id = 0
        while cap.isOpened():
            self.annotation_line_index += 1
            self.video_frame_id += 1

            is_ok, frame = cap.read()
            if not is_ok:
                break

            processed_frame = self.process_frame(frame)
            self.process_image_label_pair(
                processed_frame,
                file_name,
                annotations,
                INPUT_IMAGE_WIDTH,
                INPUT_IMAGE_HEIGHT,
            )

        cap.release()

    @abstractmethod
    def process_image_label_pair(
        self,
        frame,
        file_name,
        annotations,
        INPUT_IMAGE_WIDTH,
        INPUT_IMAGE_HEIGHT,
    ):
        """
        To process the image and label from the specific dataset
        """
        raise NotImplementedError

    def process_frame(self, frame):
        """
        Applies the transforms that need to be applied before the frame is
        saved

        Args:
            frame (numpy array): The frame that needs to be processed

        Returns:
            pytorch tensor: The processed frame
        """
        im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output_image_tensor = self.RESIZE_TRANSFORM(im_pil)

        return output_image_tensor

    def save_image_label_pair(self, file_name, output_image_tensor, label):
        """
        Saves the image and label pair to disk

        Args:
            file_name (String):
                Name to be used for the image and the label files
            output_image_tensor (pytorch tensor):
                The image that will be saved to disk
            label (List):
                The label that will be saved to disk
        """
        output_file_name = file_name + "_" + str(self.video_frame_id).zfill(4)

        output_frame = tensor_to_opencv_image(output_image_tensor)
        video_output_file_path = os.path.join(
            self.OUTPUT_IMAGES_PATH,
            output_file_name + "." + IMAGES_FILE_EXTENSION,
        )
        cv2.imwrite(video_output_file_path, output_frame)

        label_output_file_path = os.path.join(
            self.OUTPUT_LABELS_PATH, output_file_name + ".bin"
        )
        pickle.dump(label, open(label_output_file_path, "wb"))
