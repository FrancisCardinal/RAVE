import os
import argparse

import torch
import numpy as np 
import cv2
import random

from RAVE.common import Trainer
from RAVE.common.image_utils import tensor_to_opencv_image, inverse_normalize

from RAVE.eye_tracker.EyeTrackerDataset import EyeTrackerDataset, EyeTrackerInferenceDataset

from RAVE.eye_tracker.EyeTrackerDatasetBuilder import EyeTrackerDatasetBuilder
from RAVE.eye_tracker.EyeTrackerSyntheticDatasetBuilder import EyeTrackerSyntheticDatasetBuilder

from RAVE.eye_tracker.EyeTrackerModel import EyeTrackerModel
from RAVE.eye_tracker.ellipse_util import (
    ellipse_loss_function,
    draw_ellipse_on_image,
)

from RAVE.eye_tracker.GazeInferer.GazeInferer import GazeInferer


def main(TRAIN, NB_EPOCHS, CONTINUE_TRAINING, DISPLAY_VALIDATION, TEST, INFERENCE, GPU_INDEX, lr=1e-3):
    """main function of the module

    Args:
        TRAIN (bool): Whether to train the model or not
        NB_EPOCHS (int):
            Number of epochs for which to train the network(ignored if TRAIN is
            set to false)
        CONTINUE_TRAINING(bool):
            Whether to continue the training from the
            checkpoint on disk or not
        DISPLAY_VALIDATION (bool):
            Whether to display the predictions on the validation dataset or not
        TEST (bool):
            Whether to display the predictions on the test dataset or not
    """
    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda:{}".format(GPU_INDEX)

    EyeTrackerSyntheticDatasetBuilder.create_images_datasets_with_synthetic_images()

    BATCH_SIZE = 128
    training_sub_dataset = EyeTrackerDataset.get_training_sub_dataset()
    validation_sub_dataset = EyeTrackerDataset.get_validation_sub_dataset()

    training_loader = torch.utils.data.DataLoader(
        training_sub_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_sub_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    eye_tracker_model = EyeTrackerModel()
    eye_tracker_model.to(DEVICE)
    print(eye_tracker_model)

    min_validation_loss = float('inf')
    if TRAIN:
        optimizer = torch.optim.SGD(
            eye_tracker_model.parameters(),
            lr=lr,
            weight_decay=1e-4,
            momentum=0.9,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        trainer = Trainer(
            training_loader,
            validation_loader,
            ellipse_loss_function,
            DEVICE,
            eye_tracker_model,
            optimizer,
            scheduler,
            EyeTrackerDataset.EYE_TRACKER_DIR_PATH,
            CONTINUE_TRAINING,
        )

        min_validation_loss = trainer.train_with_validation(NB_EPOCHS)

    Trainer.load_best_model(
        eye_tracker_model, EyeTrackerDataset.EYE_TRACKER_DIR_PATH
    )

    if DISPLAY_VALIDATION:
        visualize_predictions(eye_tracker_model, validation_loader, DEVICE)

    if TEST:
        test_sub_dataset = EyeTrackerDataset.get_test_sub_dataset()
        test_loader = torch.utils.data.DataLoader(
            test_sub_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )
        visualize_predictions(eye_tracker_model, test_loader, DEVICE)
    
    if INFERENCE:
        inference(eye_tracker_model, DEVICE)
    
    return min_validation_loss


def visualize_predictions(model, data_loader, DEVICE):
    """Used to visualize the target and the predictions of the model on some
       input images

    Args:
        model (Module): The model used to perform the predictions
        data_loader (Dataloader):
            The dataloader that provides the images and the targets
        DEVICE (String): Device on which to perform the computations
    """
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            predictions = model(images)
            for image, prediction, label in zip(images, predictions, labels):
                image = inverse_normalize(
                    image,
                    EyeTrackerDataset.TRAINING_MEAN,
                    EyeTrackerDataset.TRAINING_STD,
                )
                image = tensor_to_opencv_image(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = draw_ellipse_on_image(image, label, color=(0, 255, 0))
                image = draw_ellipse_on_image(
                    image, prediction, color=(255, 0, 0)
                )

                cv2.imshow("validation", image)
                cv2.waitKey(1500)

def inference(model, device):
    eyeTracker_calibration_dataset = EyeTrackerInferenceDataset(os.path.join("calibration"), False)
    calibration_loader = torch.utils.data.DataLoader(
        eyeTracker_calibration_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
    )
    gaze_inferer = GazeInferer(model, calibration_loader, device)
    gaze_inferer.fit()

    eyeTracker_conversation_dataset = EyeTrackerInferenceDataset(os.path.join("conversation"), False)
    conversation_loader = torch.utils.data.DataLoader(
        eyeTracker_conversation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    gaze_inferer = GazeInferer(model, conversation_loader, device)
    gaze_inferer.infer()


def grid_search():
    lrs = [1e-2, 1e-3, 1e-4, 1e-5]  
    best_val = float('inf')
    best_lr = None 
    for lr in lrs : 
        current_val = main(
        True,
        150,
        False,
        False,
        False,
        False,
        1,
        lr
        )
        if(current_val < best_val):
            best_val = current_val 
            best_lr = lr 
            print('new best validation loss of {} was reached with {} lr'.format(best_val, best_lr))
    print('Overall best validation loss of {} was reached with {} lr'.format(best_val, best_lr))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--train", action="store_true", help="Train the neural network"
    )
    parser.add_argument(
        "-e",
        "--nb_epochs",
        action="store",
        type=int,
        default=20,
        help="Number of epoch for which to train the neural network",
    )
    parser.add_argument(
        "-c",
        "--continue_training_from_checkpoint",
        action="store_true",
        help="Continue training from checkpoint",
    )

    parser.add_argument(
        "-v",
        "--display_validation",
        action="store_true",
        help=(
            "Display the predictions of the neural network on the validation"
            "dataset"
        ),
    )
    parser.add_argument(
        "-p",
        "--predict",
        action="store_true",
        help=(
            "Display the predictions of the neural network on the test"
            "dataset"
        ),
    )

    parser.add_argument(
        "-i",
        "--inference",
        action="store_true",
        help=(
            "Runs the network in inference mode, that is, the network"
            "outputs predictions on the images of a video on disk or "
            "on a real time video feed."
        ),
    )

    parser.add_argument(
        "-g",
        "--gpu_index",
        action="store",
        type=int,
        default=0,
        help="Index of the GPU device to use",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(0)
    random.seed(42)

    #grid_search()
     
    main(
        args.train,
        args.nb_epochs,
        args.continue_training_from_checkpoint,
        args.display_validation,
        args.predict,
        args.inference,
        args.gpu_index,
    )
