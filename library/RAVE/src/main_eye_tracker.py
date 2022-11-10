import os
import argparse

import torch
import numpy as np
import cv2
import random

from RAVE.eye_tracker.EyeTrackerTrainer import EyeTrackerTrainer
from RAVE.common.image_utils import tensor_to_opencv_image, inverse_normalize

from RAVE.eye_tracker.EyeTrackerDataset import (
    EyeTrackerDataset,
    EyeTrackerFilm,
)

from RAVE.eye_tracker.EllipseAnnotationTool import EllipseAnnotationTool
from RAVE.eye_tracker.EyeTrackerDatasetBuilder import EyeTrackerDatasetBuilder

from RAVE.eye_tracker.EyeTrackerModel import EyeTrackerModel
from RAVE.eye_tracker.ellipse_util import (
    ellipse_loss_function,
    draw_ellipse_on_image,
)


def main(
    TRAIN,
    NB_EPOCHS,
    CONTINUE_TRAINING,
    DISPLAY_VALIDATION,
    TEST,
    ANNOTATE,
    FILM,
    GPU_INDEX,
    lr=5e-4,
):
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

    if FILM:
        film(EyeTrackerDataset.EYE_TRACKER_DIR_PATH)

    if ANNOTATE:
        annotate(EyeTrackerDataset.EYE_TRACKER_DIR_PATH)

    created_real_dataset = EyeTrackerDatasetBuilder.create_datasets("real_dataset")
    if created_real_dataset:
        EyeTrackerDatasetBuilder.create_datasets("old_real_dataset", True)

    BATCH_SIZE = 128
    training_sub_dataset = EyeTrackerDataset.get_training_sub_dataset()
    validation_sub_dataset = EyeTrackerDataset.get_validation_sub_dataset()

    training_loader = torch.utils.data.DataLoader(
        training_sub_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_sub_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )

    eye_tracker_model = EyeTrackerModel()
    eye_tracker_model.to(DEVICE)
    print(eye_tracker_model)

    min_validation_loss = float("inf")
    if TRAIN:
        optimizer = torch.optim.AdamW(
            eye_tracker_model.parameters(),
            lr=lr,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        trainer = EyeTrackerTrainer(
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

    EyeTrackerTrainer.load_best_model(eye_tracker_model, EyeTrackerDataset.EYE_TRACKER_DIR_PATH, DEVICE)

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
        with torch.no_grad():
            test_loss, number_of_images = 0, 0
            for images, labels, visibilities in test_loader:
                images, labels, visibilities = (
                    images.to(DEVICE),
                    labels.to(DEVICE),
                    visibilities.to(DEVICE),
                )

                # Forward Pass
                predictions, predicted_visibilities = eye_tracker_model(images)
                # Find the Loss
                predicted_pupil_are_visibles = predicted_visibilities > 0.90
                predictions = predictions * predicted_pupil_are_visibles.float()
                loss = ellipse_loss_function(predictions, labels)
                # Calculate Loss
                test_loss += loss.item()
                number_of_images += len(images)

        print("test loss = {}".format(test_loss / number_of_images))

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
        for images, labels, visibilities in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            predictions, predicted_visibilities = model(images)
            for (image, prediction, label, visibility, predicted_visibility,) in zip(
                images,
                predictions,
                labels,
                visibilities,
                predicted_visibilities,
            ):
                image = inverse_normalize(
                    image,
                    EyeTrackerDataset.TRAINING_MEAN,
                    EyeTrackerDataset.TRAINING_STD,
                )
                image = tensor_to_opencv_image(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if visibility:
                    image = draw_ellipse_on_image(image, label, color=(0, 255, 0))
                if predicted_visibility > 0.90:
                    image = draw_ellipse_on_image(image, prediction, color=(255, 0, 0))

                cv2.imshow("validation", image)
                cv2.waitKey(1500)


def film(root):
    """Films a video (most likely a video that will be part of the dataset
       once annoted)

    Args:
        root (string): Root path of the eye tracker module
    """
    camera_dataset = EyeTrackerFilm(1)
    camera_loader = torch.utils.data.DataLoader(
        camera_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    frame_height = EyeTrackerFilm.ACQUISITION_HEIGHT
    frame_width = EyeTrackerFilm.ACQUISITION_WIDTH

    file_name = input("Enter file name : ")

    output_path = os.path.join(root, EllipseAnnotationTool.WORKING_DIR, "videos", file_name + ".avi")
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        30,
        (frame_width, frame_height),
    )
    should_run = True
    while should_run:
        with torch.no_grad():
            for frames in camera_loader:
                frame = frames[0].cpu().numpy()

                out.write(frame)

                cv2.imshow("video", frame)
                key = cv2.waitKey(1)

                if key == ord("q"):
                    should_run = False

    out.release()


def annotate(root):
    """Annotates videos so that they can be part of the dataset

    Args:
        root (string): Root path of the eye tracker module
    """
    ellipse_annotation_tool = EllipseAnnotationTool(root)
    ellipse_annotation_tool.annotate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action="store_true", help="Train the neural network")
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
        help=("Display the predictions of the neural network on the validation" "dataset"),
    )
    parser.add_argument(
        "-p",
        "--predict",
        action="store_true",
        help=("Display the predictions of the neural network on the test" "dataset"),
    )

    parser.add_argument(
        "-a",
        "--annotate",
        action="store_true",
        help=("Runs the annotation tool."),
    )

    parser.add_argument(
        "-f",
        "--film",
        action="store_true",
        help=("Uses the camera to film a video for the dataset."),
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

    main(
        args.train,
        args.nb_epochs,
        args.continue_training_from_checkpoint,
        args.display_validation,
        args.predict,
        args.annotate,
        args.film,
        args.gpu_index,
    )
