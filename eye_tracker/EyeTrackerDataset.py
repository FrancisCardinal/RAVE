import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import uuid
import random 

import cv2
import pickle

IMAGE_DIMENSIONS = (128, 96)
TRANSFORM = transforms.Compose([
                                transforms.Resize(IMAGE_DIMENSIONS), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                transforms.Grayscale(),
                                transforms.ToTensor()
                                ]) 

class EyeTrackerDataset(Dataset):
    DATASET_DIR = 'dataset'

    IMAGES_DIR = 'images'
    LABELS_DIR = 'labels'

    TRAINING_DIR   = 'training'
    VALIDATION_DIR = 'validation'
    TEST_DIR       = 'test'

    IMAGES_FILE_EXTENSION = 'png'

    def __init__(self, 
                 sub_dataset_dir, 
                 transform):

        self.transform = transform
        ROOT_PATH = os.getcwd()
        BASE_PATH   = os.path.join(ROOT_PATH, EyeTrackerDataset.DATASET_DIR, sub_dataset_dir)
        self.IMAGES_DIR_PATH = os.path.join(BASE_PATH, EyeTrackerDataset.IMAGES_DIR)
        self.LABELS_DIR_PATH = os.path.join(BASE_PATH, EyeTrackerDataset.LABELS_DIR)

        self.images_paths = EyeTrackerDataset.get_multiple_workers_safe_list_of_paths(self.IMAGES_DIR_PATH)
        self.labels_paths = EyeTrackerDataset.get_multiple_workers_safe_list_of_paths(self.LABELS_DIR_PATH)


    def __len__(self):
        return len(self.images_paths)


    def __getitem__(self, idx):
        image_path, label_path = self.images_paths[idx], self.labels_paths[idx]

        image = cv2.imread(image_path)
        if self.transform:
            image = self.transform(image)

        label = pickle.load( open( label_path, "rb" ) )
        label = torch.tensor(label)

        return image, label


    def get_multiple_workers_safe_list_of_paths(directory):
        #Pour régler un problème de "fuite de mémoire" lorsqu'on a plusieurs workers pour le dataloader : https://gist.github.com/mprostock/2850f3cd465155689052f0fa3a177a50
        paths = os.listdir(directory)
        random.shuffle(paths) # Pour éviter que, plus loin, les éléments d'une même batch soient constitués d'éléments trop semblables.
        return np.array([str(uuid.uuid4()) for i in paths], dtype=np.string) 
    

    def get_training_and_validation_sub_datasets(transform = TRANSFORM):
        return EyeTrackerDataset(EyeTrackerDataset.TRAINING_DIR, transform), EyeTrackerDataset(EyeTrackerDataset.VALIDATION_DIR, transform)

    def get_test_sub_dataset(transform = TRANSFORM):
        return EyeTrackerDataset(EyeTrackerDataset.TEST_DIR, transform)