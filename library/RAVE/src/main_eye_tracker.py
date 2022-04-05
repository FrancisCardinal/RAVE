import os
import argparse

import torch
import numpy as np
import cv2
import random
import time

from RAVE.common.DANNTrainer import DANNTrainer
from RAVE.common.image_utils import tensor_to_opencv_image, inverse_normalize

from RAVE.eye_tracker.EyeTrackerDataset import (
    EyeTrackerDataset,
    EyeTrackerInferenceDataset,
)

from RAVE.eye_tracker.EllipseAnnotationTool import EllipseAnnotationTool
from RAVE.eye_tracker.EyeTrackerDatasetBuilder import EyeTrackerDatasetBuilder
from RAVE.eye_tracker.EyeTrackerSyntheticDatasetBuilder import (
    EyeTrackerSyntheticDatasetBuilder,
)

from RAVE.eye_tracker.EyeTrackerModel import EyeTrackerModel
from RAVE.eye_tracker.ellipse_util import (
    ellipse_loss_function,
    draw_ellipse_on_image,
)

from RAVE.eye_tracker.GazeInferer.GazeInfererManager import GazeInfererManager
from RAVE.face_detection.Direction2Pixel import Direction2Pixel


def main(
    TRAIN,
    NB_EPOCHS,
    CONTINUE_TRAINING,
    DISPLAY_VALIDATION,
    TEST,
    INFERENCE,
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

    created_real_dataset = EyeTrackerDatasetBuilder.create_datasets()
    if created_real_dataset:
        EyeTrackerSyntheticDatasetBuilder.create_datasets(True)

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
        optimizer = torch.optim.AdamW(eye_tracker_model.parameters(), lr=lr,)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        trainer = DANNTrainer(
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

    DANNTrainer.load_best_model(
        eye_tracker_model, EyeTrackerDataset.EYE_TRACKER_DIR_PATH, DEVICE
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
        with torch.no_grad():
            test_loss, number_of_images = 0, 0
            for images, labels, _ in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                # Forward Pass
                predictions, _ = eye_tracker_model(images)
                # Find the Loss
                loss = ellipse_loss_function(predictions, labels)
                # Calculate Loss
                test_loss += loss.item()
                number_of_images += len(images)

        print("test loss = {}".format(test_loss / number_of_images))

    if INFERENCE:
        inference(DEVICE)

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
        for images, labels, _ in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            predictions, _ = model(images)
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


def film(root):
    camera_dataset = EyeTrackerInferenceDataset(2)
    camera_loader = torch.utils.data.DataLoader(
        camera_dataset, batch_size=1, shuffle=False, num_workers=0,
    )
    frame_height = EyeTrackerDataset.IMAGE_DIMENSIONS[1]
    frame_width = EyeTrackerDataset.IMAGE_DIMENSIONS[2]

    file_name = input("Enter file name : ")

    output_path = os.path.join(
        root, EllipseAnnotationTool.WORKING_DIR, "videos", file_name + ".avi"
    )
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        30,
        (frame_width, frame_height),
    )
    should_run = True
    while should_run:
        with torch.no_grad():
            for images in camera_loader:
                image = inverse_normalize(
                    images[0],
                    EyeTrackerDataset.TRAINING_MEAN,
                    EyeTrackerDataset.TRAINING_STD,
                )
                image = tensor_to_opencv_image(image)

                out.write(image)

                cv2.imshow("video", image)
                key = cv2.waitKey(1)

                if key == ord("q"):
                    should_run = False

    out.release()


def annotate(root):
    ellipse_annotation_tool = EllipseAnnotationTool(root)
    ellipse_annotation_tool.annotate()


def inference(device):
    gaze_inferer_manager = GazeInfererManager(2, device)
    head_camera = cv2.VideoCapture(4)
    head_camera.set(cv2.CAP_PROP_FPS, 30.0)

    out = cv2.VideoWriter(
        "head_camera.avi",
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        30,
        (640, 480),
    )
    wait_for_enter("start calibration")
    gaze_inferer_manager.start_calibration_thread()

    wait_for_enter("end calibration")
    gaze_inferer_manager.end_calibration_thread()

    wait_for_enter("set offset")
    gaze_inferer_manager.set_offset("Jacob_2")

    wait_for_enter("start inference")
    gaze_inferer_manager.set_selected_calibration_path("Jacob_2")
    gaze_inferer_manager.start_inference_thread()

    FPS = 30.0
    direction_2_pixel = Direction2Pixel(-16, 21)
    for i in range(int(60 * FPS)):
        ret, frame = head_camera.read()
        time.sleep(1 / FPS)
        angle_x, angle_y = gaze_inferer_manager.get_current_gaze()
        x, y = 0, 0
        if angle_x is not None:
            print("angle_x = {} | angle_y = {}".format(angle_x, angle_y))
            x, y = direction_2_pixel.get_pixel(angle_x, angle_y)

        if ret:
            frame = cv2.flip(frame, 0)
            frame = cv2.flip(frame, 1)
            cv2.drawMarker(frame, (x, y), color=(0, 0, 255), thickness=2)

            out.write(frame)

            cv2.imshow("Facial camera", frame)
            cv2.waitKey(1)

    gaze_inferer_manager.stop_inference()
    gaze_inferer_manager.end()
    out.release()


def wait_for_enter(msg=""):
    is_waiting_for_enter = True
    while is_waiting_for_enter:
        key = input("Waiting for enter key to {}".format(msg))
        if key == "":
            is_waiting_for_enter = False


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
        args.inference,
        args.annotate,
        args.film,
        args.gpu_index,
    )
