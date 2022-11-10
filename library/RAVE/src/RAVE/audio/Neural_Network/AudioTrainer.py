from time import sleep
import torch
import torchaudio
from torchmetrics.audio import SignalNoiseRatio, SignalDistortionRatio
from tqdm import tqdm
from RAVE.common.Trainer import Trainer
import numpy as np
import matplotlib.pyplot as plt
import math

class AudioTrainer(Trainer):
    """
        Trainer class, used to train neural networks

        Args:
            training_loader (Dataloader):
                Dataloader that returns images and labels pairs of the training
                dataset
            validation_loader (Dataloader):
                Dataloader that returns images and labels pairs of the
                validation dataset
            loss_function (Functor):
                Loss function used to compute the training and
                validation losses
            device (String):
                Device on which to perform the computations
            model (Module):
                Neural network to be trained
            optimizer (Optimizer):
                Optimizer used to update the weights during training
            scheduler (_LRScheduler):
                Learning rate scheduler
            CONTINUE_TRAINING (bool):
                Whether to continue the training from the checkpoint
                on disk or not
        """
    def __init__(self, training_loader, validation_loader, loss_function, device, model, optimizer, scheduler, ROOT_DIR_PATH, CONTINUE_TRAINING, MODEL_INFO_FILE_NAME):
        super().__init__(training_loader, validation_loader, loss_function, device, model, optimizer, scheduler, ROOT_DIR_PATH, CONTINUE_TRAINING, MODEL_INFO_FILE_NAME)
        self.sdr = SignalDistortionRatio().to(device)

    def compute_training_loss(self):
        """
        Compute the training loss for the current epoch

        Returns:
            float: The training loss
        """
        self.model.train()

        training_loss = 0.0
        number_of_images = 0
        for images, labels, energy, a, b in tqdm(
                self.training_loader, "training", leave=False
        ):
            images, labels, energy = images.to(self.device), labels.to(self.device), energy.to(self.device)

            #self.show_input_signals(images)

            # Clear the gradients
            self.optimizer.zero_grad()
            # Forward Pass
            predictions, _ = self.model(images)
            # Find the Loss
            #loss = self.loss_function(predictions*total_energy, labels*total_energy)
            loss = self.loss_function(predictions*energy, labels*energy)
            #loss = self.loss_function(predictions, labels)
            # Calculate gradients
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)

            # Update Weights
            self.optimizer.step()
            # Calculate Loss
            training_loss += loss.item()
            number_of_images += len(images)

        return training_loss / (number_of_images * labels.shape[1] * labels.shape[2])

    @staticmethod
    def sqrt_hann_window(window_length, periodic=True, dtype=None, layout=torch.strided, device=None,
                         requires_grad=False):
        return torch.sqrt(torch.hann_window(window_length, periodic=periodic, dtype=dtype, layout=layout, device=device,
                                            requires_grad=requires_grad))

    def show_input_signals(self, signals):
        y2, x2 = np.mgrid[slice(0, 514, 1),
                          slice(0, 2, 2 / math.floor((32000 / 256) + 1))]


        plt.pcolormesh(x2, y2, signals[0,0,:, :].cpu().float(), shading='gouraud')
        plt.show()

    def compute_validation_loss(self):
        """
        Compute the validation loss for the current epoch

        Returns:
            float: The validation loss
        """
        with torch.no_grad():
            self.model.eval()

            validation_loss = 0.0
            number_of_images = 0
            for images, labels, energy, original_clean_signals, original_audio_freq  in tqdm(
                self.validation_loader, "validation", leave=False
            ):
                images, labels, energy, original_clean_signals, original_audio_freq = images.to(self.device), labels.to(self.device), energy.to(self.device), original_clean_signals.to(self.device), original_audio_freq.to(self.device)

                # Forward Pass
                predictions, _ = self.model(images)
                # Find the Loss
                #loss = self.loss_function(predictions*total_energy, labels*total_energy)
                loss = self.loss_function(predictions * energy, labels * energy)
                #loss = self.loss_function(predictions, labels)
                # Calculate Loss
                validation_loss += loss.item()
                number_of_images += len(images)

            before_signal_freq = original_audio_freq[:, 0, :, :] #* (1 - predictions)
            cleaned_signal_freq = original_audio_freq[:,0,:,:] * (1-predictions)
            waveform = torchaudio.transforms.InverseSpectrogram(
                n_fft=512,
                hop_length=256,
                window_fn= self.sqrt_hann_window
            ).to(self.device)

            signal_before = waveform(before_signal_freq)
            clean_signal = waveform(cleaned_signal_freq)

            beforeSdr = self.sdr(signal_before, original_clean_signals[:, 0, :]).item()
            afterSdr = self.sdr(clean_signal, original_clean_signals[:, 0, :]).item()
            print("Avant: ", beforeSdr)
            print("Après: ", afterSdr)

            return validation_loss / (number_of_images * labels.shape[1] * labels.shape[2])

