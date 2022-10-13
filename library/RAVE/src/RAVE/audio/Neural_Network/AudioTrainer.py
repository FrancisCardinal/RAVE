from time import sleep
import torch
import torchaudio
from torchmetrics.audio import SignalNoiseRatio, SignalDistortionRatio
from tqdm import tqdm
from RAVE.common.Trainer import Trainer


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
    def __init__(self, training_loader, validation_loader, loss_function, device, model, optimizer, scheduler, ROOT_DIR_PATH, CONTINUE_TRAINING):
        super().__init__(training_loader, validation_loader, loss_function, device, model, optimizer, scheduler, ROOT_DIR_PATH, CONTINUE_TRAINING)
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
        for images, labels, total_energy, _, _ in tqdm(
                self.training_loader, "training", leave=False
        ):
            images, labels, total_energy = images.to(self.device), labels.to(self.device), total_energy.to(self.device)

            # Clear the gradients
            self.optimizer.zero_grad()
            # Forward Pass
            predictions, _ = self.model(images)
            # Find the Loss
            #loss = self.loss_function(predictions*total_energy, labels*total_energy)
            energy = torch.squeeze(images[:,:,:513,:])
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
            for images, labels, total_energy, original_clean_signals, original_audio_freq  in tqdm(
                self.validation_loader, "validation", leave=False
            ):
                images, labels, total_energy, original_clean_signals, original_audio_freq = images.to(self.device), labels.to(self.device), total_energy.to(self.device), original_clean_signals.to(self.device), original_audio_freq.to(self.device)

                # Forward Pass
                predictions, _ = self.model(images)
                # Find the Loss
                #loss = self.loss_function(predictions*total_energy, labels*total_energy)
                energy = torch.squeeze(images[:, :, :513, :])
                loss = self.loss_function(predictions * energy, labels * energy)
                #loss = self.loss_function(predictions, labels)
                # Calculate Loss
                validation_loss += loss.item()
                number_of_images += len(images)

            cleaned_signal_freq = original_audio_freq[:,0,:,:] * (1-predictions)
            waveform = torchaudio.transforms.InverseSpectrogram(
                n_fft=1024,
                hop_length=256,
            ).to(self.device)
            clean_signal = waveform(cleaned_signal_freq)


            sdr = self.sdr(clean_signal, original_clean_signals[:, 0, :]).item()
            print(sdr)

            return validation_loss / (number_of_images * labels.shape[1] * labels.shape[2])