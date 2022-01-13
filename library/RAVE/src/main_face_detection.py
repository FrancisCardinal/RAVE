import numpy as np
import torch
import cv2
import argparse
from tqdm import tqdm

from RAVE.common.image_utils import (
    tensor_to_opencv_image,
    inverse_normalize,
    xyxy2xywh,
    opencv_image_to_tensor,
    scale_coords,
    scale_coords_landmarks,
)


from RAVE.face_detection.FaceDetectionDataset import FaceDetectionDataset
from RAVE.face_detection.FaceDetectionModel import FaceDetectionModel
from RAVE.common.fpsHelper import FPS

CONFIDENCE_THRESHOLD = 0.5
INTERSECTION_OVER_UNION_THRESHOLD = 0.5

# Calibration
K_640 = np.array(
    [[376.96798, 0.0, 314.09011], [0.0, 374.08737, 250.37452], [0.0, 0.0, 1.0]]
)
D_640 = np.array([-0.321459, 0.073634, -0.005091, 0.001433, 0.000000])

K_1280 = np.array(
    [[688.98267, 0.0, 654.13648], [0.0, 687.22945, 487.61611], [0.0, 0.0, 1.0]]
)
D_1280 = np.array([-0.299440, 0.074629, 0.000560, -0.001228, 0.000000])


def main(
    TRAIN, NB_EPOCHS, CONTINUE_TRAINING, DISPLAY_VALIDATION, TEST, STREAM
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
        STREAM (bool):
            Whether to display the predictions on a camera stream or not.
    """
    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"

    BATCH_SIZE = 32
    validation_sub_dataset = FaceDetectionDataset.get_validation_sub_dataset()

    validation_loader = torch.utils.data.DataLoader(
        validation_sub_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    face_detection_model = FaceDetectionModel(DEVICE)
    face_detection_model.to(DEVICE)
    # print(face_detection_model)

    if TRAIN:
        # If one day we want to train the model
        print("Train")

    if DISPLAY_VALIDATION:
        visualize_predictions(face_detection_model, validation_loader, DEVICE)

    if TEST:
        # If one day we want to test the model for training
        print("Test validation")

    if STREAM:
        visualize_predictions_on_stream(face_detection_model, DEVICE, 0)


def collate_fn(batch):
    """
    Function to override the behavior of PyTorch Dataloader, so we can have
    variable label length (The number of faces varies between images)

    Args:
        batch (ndarray): Batch with the images, labels and labels shape
    """
    img, label, label_shape = zip(*batch)
    return torch.stack(img, 0), torch.cat(label, 0), label_shape


def visualize_predictions(model, data_loader, DEVICE):
    """
    Used to visualize the target and the predictions of the model on some
    input images

    Args:
        model (Module): The model used to perform the predictions
        data_loader (Dataloader):
            The dataloader that provides the images and the targets
        DEVICE (String): Device on which to perform the computations
    """
    with torch.no_grad():
        key_pressed = None
        for batch_images, _, _ in data_loader:
            batch_images = batch_images.to(DEVICE)
            batch_predictions = model(batch_images)
            for predictions, image in zip(batch_predictions, batch_images):
                image = inverse_normalize(
                    image,
                    FaceDetectionDataset.TRAINING_MEAN,
                    FaceDetectionDataset.TRAINING_STD,
                )
                image = tensor_to_opencv_image(image)
                # Convert RGB to BGR
                image = image[:, :, ::-1]

                for i in range(predictions.size()[0]):
                    # normalization gain whwh
                    gn = torch.tensor(image.shape)[[1, 0, 1, 0]].to(DEVICE)
                    # normalization gain landmarks
                    gn_lks = torch.tensor(image.shape)[
                        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
                    ].to(DEVICE)
                    face_box_xywh = (
                        (xyxy2xywh(predictions[i, :4].view(1, 4)) / gn)
                        .view(-1)
                        .tolist()
                    )
                    confidence = predictions[i, 4].cpu().numpy()
                    landmarks = (
                        (predictions[i, 5:15].view(1, 10) / gn_lks)
                        .view(-1)
                        .tolist()
                    )
                    image = show_results(
                        image, face_box_xywh, confidence, landmarks
                    )

                cv2.imshow("validation", image)

                # Stop if escape key is pressed
                key_pressed = cv2.waitKey(1500) & 0xFF
                if key_pressed == 27:
                    break
            if key_pressed == 27:
                break


def visualize_predictions_on_stream(
    model, DEVICE, opencv_device, fps=30, width=640, height=480
):
    """
    Used to visualize the target and the predictions of the model on some
    a camera stream

    Args:
        model (Module): The model used to perform the predictions
        DEVICE (String): Device on which to perform the computations
        opencv_device (int): Device number of the camera
        fps (int): At what framerate to capture from the camera
        width (int): Width of the image in pixels
        height (int): Height of the image in pixels
    """
    cap = cv2.VideoCapture(opencv_device)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    fps = FPS()
    with torch.no_grad():
        while True:
            _, frame = cap.read()
            # Get predictions
            tensor = opencv_image_to_tensor(frame, DEVICE)
            tensor = torch.unsqueeze(tensor, 0)
            predictions = model(tensor)[0]

            # Scale coords
            predictions[:, 5:15] = scale_coords_landmarks(
                tensor.shape[2:], predictions[:, 5:15], frame.shape
            ).round()
            predictions[:, :4] = scale_coords(
                tensor.shape[2:], predictions[:, :4], frame.shape
            ).round()

            # Draw predictions
            for i in range(predictions.size()[0]):
                gn = torch.tensor(frame.shape)[[1, 0, 1, 0]].to(
                    DEVICE
                )  # normalization gain whwh
                gn_lks = torch.tensor(frame.shape)[
                    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
                ].to(
                    DEVICE
                )  # normalization gain landmarks
                xywh = (
                    (xyxy2xywh(predictions[i, :4].view(1, 4)) / gn)
                    .view(-1)
                    .tolist()
                )
                confidence = predictions[i, 4].cpu().item()
                landmarks = (
                    (predictions[i, 5:15].view(1, 10) / gn_lks)
                    .view(-1)
                    .tolist()
                )
                frame = show_results(frame, xywh, confidence, landmarks)

            fps.incrementFps()
            final_frame = fps.writeFpsToFrame(frame)

            cv2.imshow("Facial detection", final_frame)

            # Stop if escape key is pressed
            k = cv2.waitKey(30) & 0xFF
            if k == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


def show_results(img, xywh, confidence, landmarks):
    """
    Writes the bounding box, the confidence score and the landmarks to the
    frame. For now, it only shows the result of one prediction.

    Args:
        img (ndarray):
            OpenCV image with shape (height, width, 3)
        xywh (ndarray):
            Normalized values to define the bounding box with shape
            (x, y, width height).
        confidence (float):
            Confidence score.
        landmarks (list):
            Normalized landmarks of length 10. The x and y alternate in the
            list like so: [x1, y1, x2, y2, ..., x5, y5]
    """
    height, width, _ = img.shape
    img = np.ascontiguousarray(img, dtype=np.uint8)
    line_thickness = 1 or round(0.002 * (height + width) / 2) + 1

    # Bounding box
    x1 = int(xywh[0] * width - 0.5 * xywh[2] * width)
    y1 = int(xywh[1] * height - 0.5 * xywh[3] * height)
    x2 = int(xywh[0] * width + 0.5 * xywh[2] * width)
    y2 = int(xywh[1] * height + 0.5 * xywh[3] * height)
    img = cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        thickness=line_thickness,
        lineType=cv2.LINE_AA,
    )

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
    ]

    # Landmarks
    for i in range(5):
        point_x = int(landmarks[2 * i] * width)
        point_y = int(landmarks[2 * i + 1] * height)
        img = cv2.circle(
            img, (point_x, point_y), line_thickness + 1, colors[i], -1
        )

    mouth_x = int(((landmarks[6] + landmarks[8]) * width) / 2)
    mouth_y = int(((landmarks[7] + landmarks[9]) * height) / 2)

    img = cv2.circle(
        img, (mouth_x, mouth_y), line_thickness + 1, (255, 0, 0), -1
    )

    # Confidence score
    font_thickness = max(line_thickness - 1, 1)  # font thickness
    label = f"{str(confidence)[:5]}"
    cv2.putText(
        img,
        label,
        (x1, y1 - 2),
        0,
        line_thickness / 3,
        [225, 255, 255],
        thickness=font_thickness,
        lineType=cv2.LINE_AA,
    )

    return img


def find_dataset_mean_and_std(loader):
    """
    Method to find the dataset mean and standard deviation.

    Args:
        loader (Dataloader): The Pytorch Dataloader.
    """
    nb_samples = 0
    mean = 0.0
    var = 0.0
    for data, _, _ in tqdm(loader, leave=False, desc="Mean and STD"):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        var += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    var /= nb_samples
    std = torch.sqrt(var)

    print(f"Mean: {mean}")
    print(f"Std: {std}")


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
        "-s",
        "--stream",
        action="store_true",
        help=(
            "Display the predictions of the neural network on a video stream"
        ),
    )
    args = parser.parse_args()

    main(
        args.train,
        args.nb_epochs,
        args.continue_training_from_checkpoint,
        args.display_validation,
        args.predict,
        args.stream,
    )
