from time import sleep
import torch

from tqdm import tqdm
from RAVE.common.Trainer import Trainer


class AudioTrainer(Trainer):
    def __init__(self, training_loader, validation_loader, loss_function, device, model, optimizer, scheduler, ROOT_DIR_PATH, CONTINUE_TRAINING):
        super().__init__(training_loader, validation_loader, loss_function, device, model, optimizer, scheduler, ROOT_DIR_PATH, CONTINUE_TRAINING)

    def compute_training_loss(self):
        """
        Compute the training loss for the current epoch

        Returns:
            float: The training loss
        """
        self.model.train()

        training_loss = 0.0
        number_of_images = 0
        for images, labels, total_energy in tqdm(
                self.training_loader, "training", leave=False
        ):
            images, labels, total_energy = images.to(self.device), labels.to(self.device), total_energy.to(self.device)

            # Clear the gradients
            self.optimizer.zero_grad()
            # Forward Pass
            predictions = self.model(images)
            # Find the Loss
            loss = self.loss_function(predictions*total_energy, labels*total_energy)
            # Calculate gradients
            loss.backward()
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
            for images, labels, total_energy in tqdm(
                self.validation_loader, "validation", leave=False
            ):
                images, labels, total_energy = images.to(self.device), labels.to(self.device), total_energy.to(self.device)

                # Forward Pass
                predictions = self.model(images)
                # Find the Loss
                loss = self.loss_function(predictions*total_energy, labels*total_energy)
                # Calculate Loss
                validation_loss += loss.item()
                number_of_images += len(images)

            return validation_loss / (number_of_images * labels.shape[1] * labels.shape[2])