import os 
import cv2
import numpy as np 
import pickle 
from tqdm import tqdm

OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT = 128, 96 
SOURCE_DIR = 'LPW'
DESTINATION_DIR = 'dataset'
VIDEOS_DIR = 'videos'
ANNOTATIONS_DIR = 'annotations'

IMAGES_DIR = 'images'
LABELS_DIR = 'labels'

ROOT_PATH = os.getcwd()
VIDEOS_PATH      = os.path.join(ROOT_PATH, SOURCE_DIR, VIDEOS_DIR)
ANNOTATIONS_PATH = os.path.join(ROOT_PATH, SOURCE_DIR, ANNOTATIONS_DIR)

IMAGES_PATH = os.path.join(ROOT_PATH, DESTINATION_DIR, IMAGES_DIR)
LABELS_PATH = os.path.join(ROOT_PATH, DESTINATION_DIR, LABELS_DIR)

OUTPUT_VIDEO_FILE_EXTENSION = 'png'

def create_images_dataset_with_LPW_videos():
    create_directory_if_does_not_exist(IMAGES_PATH)
    create_directory_if_does_not_exist(LABELS_PATH)

    VIDEOS_FILES_NAMES = os.listdir(VIDEOS_PATH)
    ANNOTATIONS_FILES_NAMES = os.listdir(ANNOTATIONS_PATH)
    assert( len(VIDEOS_FILES_NAMES) == len(ANNOTATIONS_FILES_NAMES) ) # Chaque vidéo devrait avoir son fichier d'annotation

    ANNOTATION_EXTENSION = '.txt'
    for video_file_name in tqdm(VIDEOS_FILES_NAMES) : 
        video_path = os.path.join(VIDEOS_PATH, video_file_name)

        file_name = os.path.splitext( os.path.basename(video_file_name) )[0] + ANNOTATION_EXTENSION
        annotations_file = open(os.path.join(ANNOTATIONS_PATH, file_name), 'r')
        annotations = annotations_file.readlines()

        create_images_dataset_with_of_one_video(file_name, video_path, annotations)


def create_directory_if_does_not_exist(path): 
    if( not os.path.isdir(path) ):
        os.makedirs(path) 


def create_images_dataset_with_of_one_video(file_name, video_path, annotations): 
    cap = cv2.VideoCapture(video_path)

    INPUT_IMAGE_WIDTH  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
    INPUT_IMAGE_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  

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

        output_frame = cv2.resize(frame, (OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT)) 
        
        video_output_file_path = os.path.join(IMAGES_PATH, output_file_name + '.' + OUTPUT_VIDEO_FILE_EXTENSION)
        cv2.imwrite(video_output_file_path, output_frame)

        h, k = center_x/INPUT_IMAGE_WIDTH, center_y/INPUT_IMAGE_HEIGHT
        a, b = ellipse_width/(2*INPUT_IMAGE_WIDTH), ellipse_height/(2*INPUT_IMAGE_HEIGHT) 
        theta = np.deg2rad(angle)

        label = [h, k, a, b, theta]

        label_output_file_path = os.path.join(LABELS_PATH, output_file_name + '.bin')
        pickle.dump( label, open( label_output_file_path, "wb" ) )

    cap.release()


if __name__ =='__main__':
    create_images_dataset_with_LPW_videos()