import os 
import cv2
from PIL import Image
import pickle 
from tqdm import tqdm
from threading import Thread

from torchvision import transforms

from NormalizedEllipse import NormalizedEllipse
from videos_and_dataset_association import TRAINING_VIDEOS, VALIDATION_VIDEOS, TEST_VIDEOS

from EyeTrackerDataset import EyeTrackerDataset, IMAGE_DIMENSIONS

from image_utils import tensor_to_opencv_image, apply_image_translation, apply_image_rotation


DATASET_DIR, IMAGES_DIR, LABELS_DIR, TRAINING_DIR, VALIDATION_DIR, TEST_DIR, IMAGES_FILE_EXTENSION = EyeTrackerDataset.DATASET_DIR ,EyeTrackerDataset.IMAGES_DIR,EyeTrackerDataset.LABELS_DIR,EyeTrackerDataset.TRAINING_DIR,EyeTrackerDataset.VALIDATION_DIR,EyeTrackerDataset.TEST_DIR, EyeTrackerDataset.IMAGES_FILE_EXTENSION

class DatasetBuilder:
    """This class builds the sub-datasets. It takes videos, extracts the frames and saves them on the disk, with the corresponding labels.
       It also applies data augmentation transforms to the training sub-dataset. 
    """
    SOURCE_DIR = 'LPW'
    VIDEOS_DIR = 'videos'
    ANNOTATIONS_DIR = 'annotations'

    ROOT_PATH = os.getcwd()
    VIDEOS_PATH      = os.path.join(ROOT_PATH, SOURCE_DIR, VIDEOS_DIR)
    ANNOTATIONS_PATH = os.path.join(ROOT_PATH, SOURCE_DIR, ANNOTATIONS_DIR)

    @staticmethod
    def create_images_datasets_with_LPW_videos():
        """Static methods ; Main method of the DatasetBuilder class. This method checks if the dataset as already been built, and builds it otherwise.
        """
        BUILDERS = DatasetBuilder.get_builders()
        if(BUILDERS == -1): 
            return 
            
        print('dataset has NOT been found on disk, creating dataset')
        threads = []
        for builder in BUILDERS : 
            thread = Thread(target=builder.create_images_of_one_video_group)
            thread.start()
            threads.append(thread)
        
        for thread in threads : 
            thread.join()


    @staticmethod
    def get_builders():
        """Static methods ; Used to get the 3 DatasetBuilder objects (one for each sub-dataset)

        Returns:
            List of DatasetBuilder: The 3 DatasetBuilder objects (one for each sub-dataset)
        """
        TRAINING_PATH   = os.path.join(DatasetBuilder.ROOT_PATH, DATASET_DIR, TRAINING_DIR)
        VALIDATION_PATH = os.path.join(DatasetBuilder.ROOT_PATH, DATASET_DIR, VALIDATION_DIR)
        TEST_PATH       = os.path.join(DatasetBuilder.ROOT_PATH, DATASET_DIR, TEST_DIR)

        if( os.path.isdir(TEST_PATH) ): 
            print('dataset found on disk')
            return -1 

        BUILDERS = [ TrainingDatasetBuilder(TRAINING_VIDEOS, TRAINING_PATH, 'training   dataset'), DatasetBuilder(VALIDATION_VIDEOS, VALIDATION_PATH, 'validation dataset'), DatasetBuilder(TEST_VIDEOS, TEST_PATH, 'test       dataset') ]
        return BUILDERS


    @staticmethod
    def create_directory_if_does_not_exist(path): 
        """Creates a directory if it does not exist on the disk 

        Args:
            path (string): The path of the directory to create
        """
        if( not os.path.isdir(path) ):
            os.makedirs(path) 


    def __init__(self, 
                VIDEOS, 
                OUTPUT_DIR_PATH, 
                log_name):
        """Constructor of the DatasetBuilder class

        Args:
            VIDEOS (List of strings): The names of the videos that belong to this sub-dataset
            OUTPUT_DIR_PATH (String): Path of the directory that will contain the images and labels pairs
            log_name (String): Name to be displayed alongside the progress bar in the terminal 
        """
        self.VIDEOS = VIDEOS
        self.log_name = log_name

        self.OUTPUT_IMAGES_PATH = os.path.join(OUTPUT_DIR_PATH, IMAGES_DIR)
        self.OUTPUT_LABELS_PATH = os.path.join(OUTPUT_DIR_PATH, LABELS_DIR)
        DatasetBuilder.create_directory_if_does_not_exist(OUTPUT_DIR_PATH)
        DatasetBuilder.create_directory_if_does_not_exist(self.OUTPUT_IMAGES_PATH)
        DatasetBuilder.create_directory_if_does_not_exist(self.OUTPUT_LABELS_PATH)

        self.RESIZE_TRANSFORM = transforms.Compose([
                            transforms.Resize(IMAGE_DIMENSIONS[1:3]), 
                            transforms.ToTensor()
                            ])


    def create_images_of_one_video_group(self):
        """Gets the info of one video, then creates the images and labels pair of the video and save them to disk
        """
        for video_file_name in tqdm(self.VIDEOS, leave=False, desc=self.log_name) : 
            video_path = os.path.join(DatasetBuilder.VIDEOS_PATH, video_file_name)

            file_name = os.path.splitext( os.path.basename(video_file_name) )[0] 
            annotations_file = open(os.path.join(DatasetBuilder.ANNOTATIONS_PATH, file_name + '.txt'), 'r')
            annotations = annotations_file.readlines()

            self.create_images_dataset_with_of_one_video(file_name, video_path, annotations)


    def create_images_dataset_with_of_one_video(self, 
                                                file_name, 
                                                video_path, 
                                                annotations): 
        """Creates the images and labels pair of the video and save them to disk

        Args:
            file_name (String): Name of the video
            video_path (String): Path to the video
            annotations (List of strings): One string per frame. Each string represents the ellipse that is present on the frame
        """
        cap = cv2.VideoCapture(video_path)

        INPUT_IMAGE_WIDTH  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
        INPUT_IMAGE_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  

        self.annotation_line_index = 0 
        self.video_frame_id = 0
        while(cap.isOpened()):
            self.annotation_line_index += 1  
            self.video_frame_id += 1 

            is_ok, frame = cap.read()
            if ( not is_ok ): 
                break 
            
            angle, center_x, center_y, ellipse_width, ellipse_height = self.parse_current_annotation(annotations)

            if(angle == -1): 
                continue # The annotation files use '-1' when the pupil is not visible on a frame. 

            self.current_ellipse = NormalizedEllipse.get_normalized_ellipse_from_opencv_ellipse(center_x, ellipse_width, center_y, ellipse_height, angle, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT)
            
            output_image_tensor = self.process_frame(frame)
            
            self.save_image_label_pair(file_name, output_image_tensor, self.current_ellipse.to_list())

        cap.release()

    
    def parse_current_annotation(self, annotations):
        """Parses the current annotation to extract the parameters of the ellipse as defined by opencv

        Args:
            annotations (List of strings): All the annotations

        Returns:
            quintuplet: The parameters of the ellipse as defined by opencv 
        """
        annotation = annotations[self.annotation_line_index].split(';')
        annotation = annotation[0:-1] #The end of the line has a ';' that must be removed
        annotation = [float(i) for i in annotation] # To go from strings to floats
        annotation_frame_id, angle, center_x, center_y, ellipse_width, ellipse_height = annotation
        assert(self.video_frame_id == annotation_frame_id) # To make sure that the current annotation really belongs to the current frame 
        
        return angle, center_x, center_y, ellipse_width, ellipse_height


    def process_frame(self, frame): 
        """Applies the transforms that need to be applied before the frame is saved

        Args:
            frame (numpy array): The frame that needs to be processed

        Returns:
            pytorch tensor: The processed frame 
        """
        im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output_image_tensor = self.RESIZE_TRANSFORM(im_pil)
        
        return output_image_tensor


    def save_image_label_pair(self, 
                              file_name, 
                              output_image_tensor, 
                              label):
        """Saves the image and label pair to disk

        Args:
            file_name (String): Name to be used for the image and the label files
            output_image_tensor (pytorch tensor): The image that will be saved to disk
            label (List): The label that will be saved to disk
        """
        output_file_name = file_name + '_' + str(self.video_frame_id).zfill(4)
        
        output_frame = tensor_to_opencv_image(output_image_tensor)
        video_output_file_path = os.path.join(self.OUTPUT_IMAGES_PATH, output_file_name + '.' + IMAGES_FILE_EXTENSION)
        cv2.imwrite(video_output_file_path, output_frame)

        label_output_file_path = os.path.join(self.OUTPUT_LABELS_PATH, output_file_name + '.bin')
        pickle.dump( label, open( label_output_file_path, "wb" ) )


class TrainingDatasetBuilder(DatasetBuilder):
    """This class inherits from DatasetBuilder. It overwrites certain methods in order to 
    do offline data augmentation.
    """
    def __init__(self, 
                VIDEOS, 
                OUTPUT_DIR_PATH, 
                log_name):
        """Constructor of the TrainingDatasetBuilder. Calls the parent constructor
           and defines the training transforms

        Args:
            VIDEOS (List of strings): The names of the videos that belong to this sub-dataset
            OUTPUT_DIR_PATH (String): Path of the directory that will contain the images and labels pairs
            log_name (String): Name to be displayed alongside the progress bar in the terminal 
        """
        super().__init__(VIDEOS, 
                         OUTPUT_DIR_PATH, 
                         log_name)

        self.TRAINING_TRANSFORM = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), # random
            transforms.GaussianBlur(3), # random
            transforms.RandomInvert(0.25) # random
            ])


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
        output_image_tensor = self.apply_translation_and_rotation(output_image_tensor)

        return output_image_tensor


    def apply_translation_and_rotation(self, output_image_tensor):
        """A data augmentation operation, translates and rotates the frame randomly and reflects that 
           change on the corresponding ellipse

        Args:
            output_image_tensor (pytorch tensor): The frame on which to apply the translation and rotation

        Returns:
            pytorch tensor: The translated and rotated frame
        """
        output_image_tensor, phi = apply_image_rotation(output_image_tensor)

        output_image_tensor, x_offset, y_offset = apply_image_translation(output_image_tensor)

        self.current_ellipse.rotate_around_image_center(phi)
        self.current_ellipse.h += x_offset
        self.current_ellipse.k += y_offset

        return output_image_tensor