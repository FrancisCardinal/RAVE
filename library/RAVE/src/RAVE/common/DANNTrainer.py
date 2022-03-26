import enum
import torch
import numpy as np
from tqdm import tqdm

from RAVE.common.Trainer import Trainer

class DANNTrainer(Trainer):
    def __init__(
        self,
        training_loader,
        validation_loader,
        loss_function,
        device,
        model,
        optimizer,
        scheduler,
        ROOT_DIR_PATH,
        CONTINUE_TRAINING,
    ):
        super().__init__(
        training_loader,
        validation_loader,
        loss_function,
        device,
        model,
        optimizer,
        scheduler,
        ROOT_DIR_PATH,
        CONTINUE_TRAINING,   
        )

        self.domain_classification_loss_function = torch.nn.BCEWithLogitsLoss()

    def compute_training_loss(self):
        """
        Compute the training loss for the current epoch

        Returns:
            float: The training loss
        """
        self.model.train()

        training_loss = 0.0
        number_of_images = 0
        len_dataloader = len(self.training_loader)
        i = 0 
        for images, labels, domains in tqdm(self.training_loader, "training", leave=False):

            p = float(i + self.epoch * len_dataloader) / self.NB_EPOCHS / len_dataloader # https://github.com/fungtion/DANN 
            alpha = 2. / (1. + np.exp(-10 * p)) - 1 #  https://github.com/fungtion/DANN 

            images, labels, domains = images.to(self.device), labels.to(self.device), domains.to(self.device)

            # Clear the gradients
            self.optimizer.zero_grad()
            # Forward Pass
            predictions, classifications = self.model(images, alpha)
            # Find the Loss
            loss = self.loss_function(predictions, labels)
            domain_classification_loss = self.domain_classification_loss_function(classifications, domains.unsqueeze(1))
            loss += domain_classification_loss
            # Calculate gradients
            loss.backward()
            # Update Weights
            self.optimizer.step()
            # Calculate Loss
            training_loss += loss.item()
            number_of_images += len(images)
            i += 1

        return training_loss / number_of_images

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
            len_dataloader = len(self.validation_loader)
            i = 0
            for images, labels, domains in tqdm(self.validation_loader, "validation", leave=False):
                p = float(i + self.epoch * len_dataloader) / self.NB_EPOCHS / len_dataloader # https://github.com/fungtion/DANN 
                alpha = 2. / (1. + np.exp(-10 * p)) - 1 #  https://github.com/fungtion/DANN 
                images, labels, domains = images.to(self.device), labels.to(self.device), domains.to(self.device)

                # Forward Pass
                predictions, classifications = self.model(images, alpha)
                # Find the Loss
                loss = self.loss_function(predictions, labels)
                domain_classification_loss = self.domain_classification_loss_function(classifications, domains.unsqueeze(1))
                loss += domain_classification_loss
                # Calculate Loss
                validation_loss += loss.item()
                number_of_images += len(images)
                i += 1

            return validation_loss / number_of_images
