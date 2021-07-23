import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import random 

from PIL import Image
import pickle

IMAGE_DIMENSIONS = (1, 224, 299)

class EyeTrackerDataset(Dataset):
    """Class that handles pairs of images and labels that are on disk
    """
    DATASET_DIR = 'dataset'

    IMAGES_DIR = 'images'
    LABELS_DIR = 'labels'

    TRAINING_DIR   = 'training'
    VALIDATION_DIR = 'validation'
    TEST_DIR       = 'test'

    IMAGES_FILE_EXTENSION = 'png'

    TRAINING_MEAN, TRAINING_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    TRANSFORM = transforms.Compose([
                            transforms.Resize(IMAGE_DIMENSIONS[1:3]), 
                            transforms.ToTensor(),
                            transforms.Normalize(mean=TRAINING_MEAN, std=TRAINING_STD)
                            ])
                            
    def __init__(self, sub_dataset_dir):
        """Constructor of the EyeTrackerDataset class

        Args:
            sub_dataset_dir (String): Name of the directory of the sub-dataset 
        """

        ROOT_PATH = os.getcwd()
        BASE_PATH   = os.path.join(ROOT_PATH, EyeTrackerDataset.DATASET_DIR, sub_dataset_dir)
        self.IMAGES_DIR_PATH = os.path.join(BASE_PATH, EyeTrackerDataset.IMAGES_DIR)
        self.LABELS_DIR_PATH = os.path.join(BASE_PATH, EyeTrackerDataset.LABELS_DIR)

        self.images_paths = EyeTrackerDataset.get_multiple_workers_safe_list_of_paths(self.IMAGES_DIR_PATH)
        

    def __len__(self):
        """Method of the Dataset class that must be overwritten by this class. 
           Used to get the number of elements in the dataset

        Returns:
            int: The number of elements in the dataset
        """
        return len(self.images_paths)


    def __getitem__(self, idx):
        """Method of the Dataset class that must be overwritten by this class. 
           Used to get an image and label pair

        Args:
            idx (int): Index of the pair to get

        Returns:
            tuple: Image and label pair
        """
        image_path = self.images_paths[idx]
        image_path = os.path.join(self.IMAGES_DIR_PATH, image_path)
        file_name = os.path.splitext( os.path.basename(image_path) )[0] 
        
        label_path = os.path.join(self.LABELS_DIR_PATH, file_name + '.bin')

        image = Image.open(image_path)
        
        image = EyeTrackerDataset.TRANSFORM(image)

        label = pickle.load( open( label_path, "rb" ) )
        label = torch.tensor(label)

        return image, label


    @staticmethod
    def get_multiple_workers_safe_list_of_paths(directory):
        """Used to build a list of paths. This method prevents a memory 
            leak that happens with list of strings and multiple dataloader workers
            https://gist.github.com/mprostock/2850f3cd465155689052f0fa3a177a50  

        Args:
            directory (String): The directory of which to get the paths of its files

        Returns:
            List: The list of paths
        """
        paths = os.listdir(directory)
        random.Random(42).shuffle(paths) # Just to make sure elements of a given batch don't look alike 
        return np.array([str(i) for i in paths], dtype=np.str) 
    
    
    @staticmethod
    def get_training_sub_dataset():
        """Used to get the training sub dataset

        Returns:
            EyeTrackerDataset: The training sub dataset
        """
        return EyeTrackerDataset(EyeTrackerDataset.TRAINING_DIR)


    @staticmethod
    def get_validation_sub_dataset():
        """Used to get the validation sub dataset

        Returns:
            EyeTrackerDataset: The validation sub dataset
        """
        return EyeTrackerDataset(EyeTrackerDataset.VALIDATION_DIR)


    @staticmethod    
    def get_test_sub_dataset():
        """Used to get the test sub dataset

        Returns:
            EyeTrackerDataset: The test sub dataset
        """
        return EyeTrackerDataset(EyeTrackerDataset.TEST_DIR)