import os
from numpy import block

import torch
import argparse
import matplotlib.pyplot as plt

from RAVE.common.Trainer import Trainer
from RAVE.audio.Neural_Network.AudioModel import AudioModel
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
        DEVICE = "cuda"
    
    # todo: call class method to create dataset if not on disk
    dataset = AudioDataset(dataset_path='/Users/felixducharmeturcotte/Documents/audioDataset', device=DEVICE)
    spect = next(iter(dataset))[0]

    
    plt.pcolormesh(spect[0].float(), shading='gouraud')
    plt.show(block=True)
    BATCH_SIZE = 128

    # todo: get training subDataset with class method
    training_sub_dataset = None
    # todo: get validation subDataset with class method
    validation_sub_dataset = None

    # todo: load training dataset
    """trainer_loader = torch.utils.data.DataLoader(
        training_sub_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )"""
    trainer_loader = None

    # todo: load testing dataset
    """validation_loader = torch.utils.data.DataLoader(
        validation_sub_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )"""
    validation_loader = None

    # todo: get directory from dataset class
    directory = os.path.join("RAVE", "audio")

    audioModel = AudioModel(input_size=16000, hidden_size=10, num_layers=1)
    audioModel.to(DEVICE)
    print(audioModel)

    if TRAIN:
        optimizer = torch.optim.SGD(
            audioModel.parameters(),
            lr=1e-03
        )
        scheduler = None

        trainer = Trainer(
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
        #trainer.train_with_validation(NB_EPOCHS)

    if DISPLAY_VALIDATION:
        print('showing validation')

    if TEST:
        print('testing')



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
    args = parser.parse_args()

    main(
        args.train,
        args.nb_epochs,
        args.continue_training_from_checkpoint,
        args.display_validation,
        args.predict,
    )