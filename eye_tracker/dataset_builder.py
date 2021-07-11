import os 
import cv2
import numpy as np 
import pickle 
from tqdm import tqdm

from videos_and_dataset_association import TRAINING_VIDEOS, VALIDATION_VIDEOS, TEST_VIDEOS

from EyeTrackerDataset import EyeTrackerDataset, IMAGE_DIMENSIONS
DATASET_DIR , IMAGES_DIR, LABELS_DIR, TRAINING_DIR, VALIDATION_DIR, TEST_DIR, IMAGES_FILE_EXTENSION = EyeTrackerDataset.DATASET_DIR ,EyeTrackerDataset.IMAGES_DIR,EyeTrackerDataset.LABELS_DIR,EyeTrackerDataset.TRAINING_DIR,EyeTrackerDataset.VALIDATION_DIR,EyeTrackerDataset.TEST_DIR, EyeTrackerDataset.IMAGES_FILE_EXTENSION

OUTPUT_IMAGE_DIMENSIONS = (IMAGE_DIMENSIONS[2], IMAGE_DIMENSIONS[1]) # opencv s'attend à recevoir (Width, Height) (i.e, le sens contraire de pytorch)
SOURCE_DIR = 'LPW'
VIDEOS_DIR = 'videos'
ANNOTATIONS_DIR = 'annotations'

ROOT_PATH = os.getcwd()
VIDEOS_PATH      = os.path.join(ROOT_PATH, SOURCE_DIR, VIDEOS_DIR)
ANNOTATIONS_PATH = os.path.join(ROOT_PATH, SOURCE_DIR, ANNOTATIONS_DIR)

TRAINING_PATH   = os.path.join(ROOT_PATH, DATASET_DIR, TRAINING_DIR)
VALIDATION_PATH = os.path.join(ROOT_PATH, DATASET_DIR, VALIDATION_DIR)
TEST_PATH       = os.path.join(ROOT_PATH, DATASET_DIR, TEST_DIR)

VIDEO_GROUPS = [[TRAINING_VIDEOS, TRAINING_PATH], [VALIDATION_VIDEOS, VALIDATION_PATH], [TEST_VIDEOS, TEST_PATH]]

def create_images_dataset_with_LPW_videos():
    ANNOTATION_EXTENSION = '.txt'

    for VIDEO_GROUP in tqdm(VIDEO_GROUPS, leave=False) : 
        VIDEOS, OUTPUT_DIR_PATH = VIDEO_GROUP
        for video_file_name in tqdm(VIDEOS, leave=False) : 
            video_path = os.path.join(VIDEOS_PATH, video_file_name)

            file_name = os.path.splitext( os.path.basename(video_file_name) )[0] 
            annotations_file = open(os.path.join(ANNOTATIONS_PATH, file_name + ANNOTATION_EXTENSION), 'r')
            annotations = annotations_file.readlines()

            create_images_dataset_with_of_one_video(file_name, video_path, annotations, OUTPUT_DIR_PATH)


def create_directory_if_does_not_exist(path): 
    if( not os.path.isdir(path) ):
        os.makedirs(path) 


def create_images_dataset_with_of_one_video(file_name, video_path, annotations, OUTPUT_DIR_PATH): 
    cap = cv2.VideoCapture(video_path)

    INPUT_IMAGE_WIDTH  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
    INPUT_IMAGE_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  

    OUTPUT_IMAGES_PATH = os.path.join(OUTPUT_DIR_PATH, IMAGES_DIR)
    OUTPUT_LABELS_PATH = os.path.join(OUTPUT_DIR_PATH, LABELS_DIR)

    create_directory_if_does_not_exist(OUTPUT_DIR_PATH)
    create_directory_if_does_not_exist(OUTPUT_IMAGES_PATH)
    create_directory_if_does_not_exist(OUTPUT_LABELS_PATH)

    annotation_line_index = 0 
    video_frame_id = 0
    while(cap.isOpened()):
        annotation_line_index += 1  
        video_frame_id += 1 

        is_ok, frame = cap.read()
        if ( not is_ok ): 
            break 
        
        annotation = annotations[annotation_line_index].split(';')
        annotation = annotation[0:-1] #le retour a la ligne est compté comme un élément, car la ligne se termine avec un ';' : il faut le retirer
        annotation = [float(i) for i in annotation] # pour passer de des strings à des floats
        annotation_frame_id, angle, center_x, center_y, ellipse_width, ellipse_height = annotation
        assert(video_frame_id == annotation_frame_id) # Pour s'assurer que l'annotation courante est belle est bien liée à la frame courante. 

        if(angle == -1): 
            continue # Les fichiers d'annotations utilisent '-1' pour toutes les valeurs lorsqu'une frame donnée ne comporte pas de pupille visible. 
        
        output_file_name = file_name + '_' + str(video_frame_id).zfill(4)

        output_frame = cv2.resize(frame, OUTPUT_IMAGE_DIMENSIONS) 
        
        video_output_file_path = os.path.join(OUTPUT_IMAGES_PATH, output_file_name + '.' + IMAGES_FILE_EXTENSION)
        cv2.imwrite(video_output_file_path, output_frame)

        h, k = center_x/INPUT_IMAGE_WIDTH, center_y/INPUT_IMAGE_HEIGHT
        a, b = ellipse_width/(2*INPUT_IMAGE_WIDTH), ellipse_height/(2*INPUT_IMAGE_HEIGHT) 
        theta = np.deg2rad(angle)

        label = [h, k, a, b, theta]

        label_output_file_path = os.path.join(OUTPUT_LABELS_PATH, output_file_name + '.bin')
        pickle.dump( label, open( label_output_file_path, "wb" ) )

    cap.release()


if __name__ =='__main__':
    create_images_dataset_with_LPW_videos()