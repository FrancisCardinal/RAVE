import multiprocessing
import os

import numpy as np
import torch
import torchaudio

import argparse
import matplotlib.pyplot as plt

from RAVE.audio.Neural_Network.AudioModel import AudioModel
from RAVE.audio.Neural_Network.AudioTrainer import AudioTrainer
from RAVE.audio.Dataset.AudioDataset import AudioDataset


def main(TRAIN, NB_EPOCHS, CONTINUE_TRAINING, DISPLAY_VALIDATION, TEST, GPU, DATASET_PATH):
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
        GPU (int):
            Which GPU number to use for training (0 or 1)
    """
    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda:" + str(GPU)

    dataset = AudioDataset(dataset_path=DATASET_PATH)

    BATCH_SIZE = 32
    lenght_dataset = len(dataset)
    validation_size = round(lenght_dataset*0.2)

    training_sub_dataset, validation_sub_dataset = torch.utils.data.random_split(dataset, [ lenght_dataset - validation_size, validation_size])

    trainer_loader = torch.utils.data.DataLoader(
        training_sub_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=25,
        pin_memory=True,
        persistent_workers=True
    )


    validation_loader = torch.utils.data.DataLoader(
        validation_sub_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=25,
        pin_memory=True,
        persistent_workers=True
    )

    # todo: get directory from dataset class
    directory = os.path.join('/home/rave/RAVE-Audio/RAVE/library/RAVE/src/RAVE/audio/Neural_Network/model')

    audioModel = AudioModel(input_size=514, hidden_size=512, num_layers=2)
    audioModel.to(DEVICE)
    print(audioModel)

    if TRAIN:
        optimizer = torch.optim.Adam(
            audioModel.parameters(),
            lr=2e-03
        )
        scheduler = None#torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True)
        trainer = AudioTrainer(
            trainer_loader,
            validation_loader,
            torch.nn.MSELoss(reduction='sum'),
            DEVICE,
            audioModel,
            optimizer,
            scheduler,
            directory,
            CONTINUE_TRAINING,
            MODEL_INFO_FILE_NAME= "saved_model.pth"
        )
        trainer.train_with_validation(NB_EPOCHS)

    AudioTrainer.load_best_model(
        audioModel, directory
    )
    if DISPLAY_VALIDATION:
        visualize_predictions(audioModel, validation_loader, DEVICE, dataset)

    if TEST:
        test(audioModel, validation_loader, DEVICE, dataset)


def visualize_predictions(model, data_loader, DEVICE, dataset):
    """Used to visualize the target and the predictions of the model on some
       input images

    Args:
        model (Module): The model used to perform the predictions
        data_loader (Dataloader):
            The dataloader that provides the images and the targets
        DEVICE (String): Device on which to perform the computations
        dataset (Dataset): Dataset used to visualize predictions
    """
    with torch.no_grad():
        for audios, labels, _ in data_loader:
            audios, labels = audios.to(DEVICE), labels.to(DEVICE)
            predictions, _ = model(audios)

            for audio, prediction, label in zip(audios, predictions, labels):
                audio = torch.squeeze(audio)
                y, x = np.mgrid[slice(0, 513, 1),
                                slice(0, dataset.duration, dataset.duration/dataset.nb_chunks)]
                y2, x2 = np.mgrid[slice(0, 1026, 1),
                                slice(0, dataset.duration, dataset.duration / dataset.nb_chunks)]

                fig, axs = plt.subplots(5)
                fig.suptitle('Vertically stacked subplots')
                pc0 = axs[0].pcolormesh(x2,y2,audio[:,:].cpu().float(), shading='gouraud')
                pc1 = axs[1].pcolormesh(x,y,prediction.cpu().float(), shading='gouraud', vmin=0, vmax=1)
                pc2 = axs[2].pcolormesh(x,y,label.cpu().float(), shading='gouraud', vmin=0, vmax=1)
                pc3 = axs[3].pcolormesh(x, y, 1- prediction.cpu().float(), shading='gouraud', vmin=0, vmax=1)
                pc4 = axs[4].pcolormesh(x, y, 1 - label.cpu().float(), shading='gouraud', vmin=0, vmax=1)
                axs[2].set_xlabel("Temps(s)")
                axs[0].set_ylabel("S-N Signal")
                axs[1].set_ylabel("N Pred")
                axs[2].set_ylabel("N Target")
                axs[3].set_ylabel("S Pred")
                axs[4].set_ylabel("S Target")

                fig.colorbar(pc0, ax=axs[0])
                fig.colorbar(pc1, ax=axs[1])
                fig.colorbar(pc2, ax=axs[2])
                fig.colorbar(pc3, ax=axs[3])
                fig.colorbar(pc4, ax=axs[4])
                plt.show()


def test(model, data_loader, DEVICE, dataset):
    """
    Used to permorm tests to generate output .wav with masks predictions from recurent neural network
    Args:
        model (Module): Neural network to be trained
        data_loader (Dataloader): Dataloader that returns signals from the test dataset
        DEVICE (string): Device on which to perform the computations
        dataset (Dataset): Dataset used to get signals

    """
    audios, labels, _ = next(iter(data_loader))
    audios, labels = audios.to(DEVICE), labels.to(DEVICE)
    index = 0
    predictions, _ = model(audios[index:index+1,:,:])

    item_path = dataset.data[index]
    audio_signal, _, _, _, _, _, _ = dataset.load_item_from_disk(
        item_path)

    audio_freq = dataset.transformation(audio_signal[:, :32000]).to(DEVICE)
    mvdr = torchaudio.transforms.MVDR(ref_channel=0, solution='ref_channel', multi_mask=False, online=True)
    istft = torchaudio.transforms.InverseSpectrogram(n_fft=1024, hop_length=256)

    stft_est = mvdr(audio_freq.type(torch.complex128), 1 - predictions, predictions)

    est = istft(stft_est.detach().cpu(), length=audios.shape[-1] * 256)

    torchaudio.save('output.wav', torch.unsqueeze(est, dim=0).float(), 16000)
    print('output.wav file generated - ' + item_path)


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

    parser.add_argument(
        "-g",
        "--gpu",
        action="store",
        type=int,
        default=0,
        help="Which GPu to use (0 or 1)",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        action="store",
        type=str,
        help="Path to the dataset Directory to train on",
    )

    args = parser.parse_args()

    main(
        args.train,
        args.nb_epochs,
        args.continue_training_from_checkpoint,
        args.display_validation,
        args.predict,
        args.gpu,
        args.dataset,
    )
