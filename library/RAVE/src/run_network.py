import multiprocessing
import os

import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt

from RAVE.audio.Neural_Network.AudioModel import AudioModel
from RAVE.audio.Neural_Network.AudioTrainer import AudioTrainer
from RAVE.audio.Dataset.AudioDataset import AudioDataset


def main(TRAIN, NB_EPOCHS, CONTINUE_TRAINING, DISPLAY_VALIDATION, TEST):
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
        DEVICE = "cuda:1"

    # training_sub_dataset = AudioDataset(dataset_path='/Users/felixducharmeturcotte/Documents/datasetV2/training', device=DEVICE)
    # validation_sub_dataset = AudioDataset(dataset_path='/Users/felixducharmeturcotte/Documents/datasetV2/validation', device=DEVICE)

    dataset = AudioDataset(dataset_path='/home/rave/audiodataset/dataset')

    BATCH_SIZE = 32
    lenght_dataset = len(dataset)
    validation_size = round(lenght_dataset*0.2)

    training_sub_dataset, validation_sub_dataset = torch.utils.data.random_split(dataset, [ lenght_dataset - validation_size, validation_size])

    trainer_loader = torch.utils.data.DataLoader(
        training_sub_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=60,
        pin_memory=True,
        persistent_workers=True
    )


    validation_loader = torch.utils.data.DataLoader(
        validation_sub_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=60,
        pin_memory=True,
        persistent_workers=True
    )

    # todo: get directory from dataset class
    directory = os.path.join(os.getcwd(), 'model')

    audioModel = AudioModel(input_size=1026, hidden_size=128, num_layers=2)
    audioModel.to(DEVICE)
    print(audioModel)

    if TRAIN:
        optimizer = torch.optim.Adam(
            audioModel.parameters(),
            lr=2e-03
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True)
        trainer = AudioTrainer(
            trainer_loader,
            validation_loader,
            torch.nn.MSELoss(reduction='sum'),
            DEVICE,
            audioModel,
            optimizer,
            scheduler,
            directory,
            CONTINUE_TRAINING
        )
        trainer.train_with_validation(NB_EPOCHS)

    AudioTrainer.load_best_model(
        audioModel, directory
    )
    if DISPLAY_VALIDATION:
        visualize_predictions(audioModel, validation_loader, DEVICE, dataset)

    if TEST:
        print('testing')


def visualize_predictions(model, data_loader, DEVICE, dataset):
    """Used to visualize the target and the predictions of the model on some
       input images

    Args:
        model (Module): The model used to perform the predictions
        data_loader (Dataloader):
            The dataloader that provides the images and the targets
        DEVICE (String): Device on which to perform the computations
    """
    with torch.no_grad():
        for audios, labels, _ in data_loader:
            audios, labels = audios.to(DEVICE), labels.to(DEVICE)
            predictions = model(audios)
            for audio, prediction, label in zip(audios, predictions, labels):
                audio = torch.squeeze(audio)
                y, x = np.mgrid[slice(0, 513, 1),
                                slice(0, dataset.duration, dataset.duration/dataset.nb_chunks)]

                fig, axs = plt.subplots(3)
                fig.suptitle('Vertically stacked subplots')
                axs[0].pcolormesh(x,y,audio[:513,:].cpu().float(), shading='gouraud')
                axs[1].pcolormesh(x,y,prediction.cpu().float(), shading='gouraud')
                axs[2].pcolormesh(x,y,label.cpu().float(), shading='gouraud')
                axs[2].set_xlabel("Temps(s)")
                axs[1].set_ylabel("Fréquences (hz)")
                plt.show()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
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
    args = parser.parse_args()

    main(
        args.train,
        args.nb_epochs,
        args.continue_training_from_checkpoint,
        args.display_validation,
        args.predict,
    )
