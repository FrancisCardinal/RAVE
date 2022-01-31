import torch
import cv2
import argparse
import os
from PIL import Image
import pickle

from RAVE.common import Trainer
from RAVE.common.image_utils import tensor_to_opencv_image, inverse_normalize

from RAVE.eye_tracker.EyeTrackerDataset import EyeTrackerDataset

from RAVE.eye_tracker.EyeTrackerDatasetBuilder import EyeTrackerDatasetBuilder
from RAVE.eye_tracker.EyeTrackerSyntheticDatasetBuilder import EyeTrackerSyntheticDatasetBuilder

from RAVE.eye_tracker.EyeTrackerModel import EyeTrackerModel
from RAVE.eye_tracker.ellipse_util import (
    ellipse_loss_function,
    draw_ellipse_on_image,
)


def main(TRAIN, NB_EPOCHS, CONTINUE_TRAINING, DISPLAY_VALIDATION, TEST, GPU_INDEX):
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

    if TRAIN:
        optimizer = torch.optim.SGD(
            eye_tracker_model.parameters(),
            lr=1e-3,
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

        trainer.train_with_validation(NB_EPOCHS)

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
        visualize_predictions_test(eye_tracker_model, test_loader, DEVICE)


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

def visualize_predictions_test(model, data_loader, device):
    with torch.no_grad():
        model.eval()

        root = '/home/rave/RAVE/library/RAVE/src/RAVE/eye_tracker/dataset/test/'

        number_of_images = 1300
        for i in range(number_of_images):
            # Forward Pass
            image_path = os.path.join(root + 'images/', str(i)+'.png')
            image = Image.open(image_path)
            image = data_loader.dataset.PRE_PROCESS_TRANSFORM(image)
            image = data_loader.dataset.NORMALIZE_TRANSFORM(image)
            image = image.unsqueeze(0) 

            prediction = model(image.to(device))
            label = pickle.load(open(root + 'labels/' + str(i) + '.bin', "rb"))
            label = torch.tensor(label["ellipse"], device=device)

            image = image.squeeze()
            image = inverse_normalize(
                image,
                EyeTrackerDataset.TRAINING_MEAN,
                EyeTrackerDataset.TRAINING_STD,
            )
            image = tensor_to_opencv_image(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            cv2.imwrite('out/images/' + str(i)+'.png', image)
            pickle.dump(prediction[0].cpu().numpy().tolist(), open('out/labels/' + str(i)+'.bin', "wb"))

            image = draw_ellipse_on_image( image, prediction[0], color=(255, 0, 0))
            #image = draw_ellipse_on_image(image, label, color=(0, 255, 0))

            cv2.imshow("test", image)
            cv2.waitKey(1)
            #cv2.waitKey(int(1000/25.0))


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
        "--gpu_index",
        action="store",
        type=int,
        default=0,
        help="Index of the GPU device to use",
    )
    args = parser.parse_args()

    main(
        args.train,
        args.nb_epochs,
        args.continue_training_from_checkpoint,
        args.display_validation,
        args.predict,
        args.gpu_index,
    )
