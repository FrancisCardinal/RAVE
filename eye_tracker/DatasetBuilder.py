import os 
import cv2
from PIL import Image
import numpy as np 
import pickle 
from tqdm import tqdm
from threading import Thread
import random

import torch
from torchvision import transforms
from torch.nn import functional as F

from videos_and_dataset_association import TRAINING_VIDEOS, VALIDATION_VIDEOS, TEST_VIDEOS

from EyeTrackerDataset import EyeTrackerDataset, IMAGE_DIMENSIONS

from image_utils import tensor_to_opencv_image

DATASET_DIR, IMAGES_DIR, LABELS_DIR, TRAINING_DIR, VALIDATION_DIR, TEST_DIR, IMAGES_FILE_EXTENSION = EyeTrackerDataset.DATASET_DIR ,EyeTrackerDataset.IMAGES_DIR,EyeTrackerDataset.LABELS_DIR,EyeTrackerDataset.TRAINING_DIR,EyeTrackerDataset.VALIDATION_DIR,EyeTrackerDataset.TEST_DIR, EyeTrackerDataset.IMAGES_FILE_EXTENSION

class DatasetBuilder:
    SOURCE_DIR = 'LPW'
    VIDEOS_DIR = 'videos'
    ANNOTATIONS_DIR = 'annotations'

    ROOT_PATH = os.getcwd()
    VIDEOS_PATH      = os.path.join(ROOT_PATH, SOURCE_DIR, VIDEOS_DIR)
    ANNOTATIONS_PATH = os.path.join(ROOT_PATH, SOURCE_DIR, ANNOTATIONS_DIR)

    @staticmethod
    def create_images_datasets_with_LPW_videos():
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
        TRAINING_PATH   = os.path.join(DatasetBuilder.ROOT_PATH, DATASET_DIR, TRAINING_DIR)
        VALIDATION_PATH = os.path.join(DatasetBuilder.ROOT_PATH, DATASET_DIR, VALIDATION_DIR)
        TEST_PATH       = os.path.join(DatasetBuilder.ROOT_PATH, DATASET_DIR, TEST_DIR)

        if( os.path.isdir(TEST_PATH) ): 
            print('dataset found on disk')
            return -1 

        BUILDERS = [ TrainingDatasetBuilder(TRAINING_VIDEOS, TRAINING_PATH), DatasetBuilder(VALIDATION_VIDEOS, VALIDATION_PATH), DatasetBuilder(TEST_VIDEOS, TEST_PATH) ]
        return BUILDERS


    @staticmethod
    def create_directory_if_does_not_exist(path): 
        if( not os.path.isdir(path) ):
            os.makedirs(path) 


    def __init__(self, VIDEOS, OUTPUT_DIR_PATH):
        self.VIDEOS = VIDEOS

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
        for video_file_name in tqdm(self.VIDEOS, leave=False) : 
            video_path = os.path.join(DatasetBuilder.VIDEOS_PATH, video_file_name)

            file_name = os.path.splitext( os.path.basename(video_file_name) )[0] 
            annotations_file = open(os.path.join(DatasetBuilder.ANNOTATIONS_PATH, file_name + '.txt'), 'r')
            annotations = annotations_file.readlines()

            self.create_images_dataset_with_of_one_video(file_name, video_path, annotations)


    def create_images_dataset_with_of_one_video(self, file_name, video_path, annotations): 
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
                continue # Les fichiers d'annotations utilisent '-1' pour toutes les valeurs lorsqu'une frame donnée ne comporte pas de pupille visible. 

            self.current_ellipse = Ellipse.get_normalized_ellipse_from_opencv_ellipse(center_x, ellipse_width, center_y, ellipse_height, angle, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT)
            
            output_image_tensor = self.process_frame(frame)
            
            self.save_image_label_pair(file_name, output_image_tensor, self.current_ellipse.to_list())

        cap.release()

    
    def parse_current_annotation(self, annotations):
        annotation = annotations[self.annotation_line_index].split(';')
        annotation = annotation[0:-1] #le retour a la ligne est compté comme un élément, car la ligne se termine avec un ';' : il faut le retirer
        annotation = [float(i) for i in annotation] # pour passer de des strings à des floats
        annotation_frame_id, angle, center_x, center_y, ellipse_width, ellipse_height = annotation
        assert(self.video_frame_id == annotation_frame_id) # Pour s'assurer que l'annotation courante est belle est bien liée à la frame courante. 
        
        return angle, center_x, center_y, ellipse_width, ellipse_height


    def process_frame(self, frame): 
        im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output_image_tensor = self.RESIZE_TRANSFORM(im_pil)
        
        return output_image_tensor


    def save_image_label_pair(self, file_name, output_image_tensor, label):
        output_file_name = file_name + '_' + str(self.video_frame_id).zfill(4)
        
        output_frame = tensor_to_opencv_image(output_image_tensor)
        video_output_file_path = os.path.join(self.OUTPUT_IMAGES_PATH, output_file_name + '.' + IMAGES_FILE_EXTENSION)
        cv2.imwrite(video_output_file_path, output_frame)

        label_output_file_path = os.path.join(self.OUTPUT_LABELS_PATH, output_file_name + '.bin')
        pickle.dump( label, open( label_output_file_path, "wb" ) )


class TrainingDatasetBuilder(DatasetBuilder):
    def __init__(self, VIDEOS, OUTPUT_DIR_PATH):
        super().__init__(VIDEOS, OUTPUT_DIR_PATH)

        self.TRAINING_TRANSFORM = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), # random
            transforms.GaussianBlur(3), # random
            transforms.RandomInvert(0.25) # random
            ])


    def process_frame(self, frame):         
        im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        output_image_tensor = self.RESIZE_TRANSFORM(im_pil)
        output_image_tensor = self.TRAINING_TRANSFORM(output_image_tensor)
        output_image_tensor = self.apply_translation(output_image_tensor)

        return output_image_tensor


    def apply_translation(self, output_image_tensor):
        x_offset = random.uniform(-0.2, 0.2)
        y_offset = random.uniform(-0.2, 0.2)

        self.current_ellipse.h += x_offset
        self.current_ellipse.k += y_offset

        transformation_matrix = torch.tensor([
            [1, 0, -x_offset*2],
            [0, 1, -y_offset*2]
        ], dtype=torch.float) # On a besoin des '*2' ici parce que affine_grid considère le coin supérieur gauche comme [-1, -1] et celui inférieur droit comme [1, 1] (contrairement à la convention de ce module où le coin supérieur gauche est [0, 0])

        grid = F.affine_grid(transformation_matrix.unsqueeze(0), output_image_tensor.unsqueeze(0).size())
        output_image_tensor = F.grid_sample(output_image_tensor.unsqueeze(0), grid)
        output_image_tensor = output_image_tensor.squeeze(0)

        return output_image_tensor


class Ellipse:
    def __init__(self, h, k, a, b, theta) :
        self.h = h
        self.k = k 
        self.a = a 
        self.b = b 
        self.theta = theta 
    

    def to_list(self):
        return [self.h, self.k, self.a, self.b, self.theta]


    @staticmethod
    def get_normalized_ellipse_from_opencv_ellipse(center_x, ellipse_width, center_y, ellipse_height, angle, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT): 
        h, k = center_x/INPUT_IMAGE_WIDTH, center_y/INPUT_IMAGE_HEIGHT
        a, b = ellipse_width/(2*INPUT_IMAGE_WIDTH), ellipse_height/(2*INPUT_IMAGE_HEIGHT) 
        theta = np.deg2rad(angle)/(2*np.pi)

        return Ellipse(h, k, a, b, theta)