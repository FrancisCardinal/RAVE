import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from time import localtime, strftime, time
from datetime import timedelta

from RAVE.common.Trainer import Trainer


class DANNTrainer(Trainer):
    """Class that implements the DANN algorithm to train a network"""

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
        """Constructor of the DANNTrainer class

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

        self.training_regression_losses, self.validation_regression_losses = (
            [],
            [],
        )
        self.training_domain_losses, self.validation_domain_losses = [], []

        self.figure, (self.ax1, self.ax2) = plt.subplots(2, 1, sharey=False)
        plt.ion()
        self.figure.show()

    def train_with_validation(self, NB_EPOCHS):
        """
        Main method of the class, used to train the model.
        Args:
            NB_EPOCHS (int): Number of epoch for which to train the model
        """
        self.NB_EPOCHS = NB_EPOCHS
        start_time = time()

        self.epoch = 0
        while (self.epoch < self.NB_EPOCHS) and (not self.terminate_training):
            (
                current_regression_training_loss,
                current_domain_training_loss,
            ) = self.compute_training_loss()
            (
                current_regression_validation_loss,
                current_domain_validation_loss,
            ) = self.compute_validation_loss()

            self.training_regression_losses.append(
                current_regression_training_loss
            )
            self.validation_regression_losses.append(
                current_regression_validation_loss
            )

            self.training_domain_losses.append(current_domain_training_loss)
            self.validation_domain_losses.append(
                current_domain_validation_loss
            )
            self.update_plot()

            epoch_stats = (
                f"Epoch {self.epoch:0>4d} | "
                f"validation_loss={current_regression_validation_loss:.6f} | "
                f"training_loss={current_regression_training_loss:.6f}"
            )

            if self.min_validation_loss > current_regression_validation_loss:
                epoch_stats = epoch_stats + (
                    "  | Min validation loss decreased("
                    f"{self.min_validation_loss:.6f}--->"
                    f"{current_regression_validation_loss:.6f})"
                    f": Saved the model"
                )
                self.min_validation_loss = current_regression_validation_loss

                self.save_model_and_training_info()

            print(epoch_stats)
            self.scheduler.step(current_regression_validation_loss)
            self.epoch += 1

        self.terminate_training = True
        min_training_loss = min(self.training_regression_losses)
        time_of_completion = strftime("%Y-%m-%d %H:%M:%S", localtime())
        ellapsed_time = str(timedelta(seconds=(time() - start_time)))
        figure_title = (
            f"{time_of_completion:s} | "
            f"ellapsed_time={ellapsed_time:s} | "
            f"min_validation_loss={self.min_validation_loss:.6f} | "
            f"min_training_loss={min_training_loss:.6f}"
        )

        print(figure_title)
        plt.savefig(
            os.path.join(
                self.ROOT_DIR_PATH,
                Trainer.TRAINING_SESSIONS_DIR,
                figure_title + ".png",
            ),
            dpi=200,
        )
        plt.close(None)
        return self.min_validation_loss

    def compute_training_loss(self):
        """
        Compute the training loss for the current epoch

        Returns:
            float: The training loss
        """
        self.model.train()

        domain_training_loss, regression_training_loss = 0.0, 0.0
        number_of_images = 0
        len_dataloader = len(self.training_loader)
        i = 0
        for images, labels, domains in tqdm(
            self.training_loader, "training", leave=False
        ):

            p = (
                float(i + self.epoch * len_dataloader)
                / self.NB_EPOCHS
                / len_dataloader
            )  # https://github.com/fungtion/DANN
            # https://github.com/fungtion/DANN
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

            images, labels, domains = (
                images.to(self.device),
                labels.to(self.device),
                domains.to(self.device),
            )

            # Clear the gradients
            self.optimizer.zero_grad()
            # Forward Pass
            predictions, classifications = self.model(images, 2 * alpha)
            # Find the Loss
            regression_loss = self.loss_function(predictions, labels)
            domain_loss = self.domain_classification_loss_function(
                classifications, domains.unsqueeze(1)
            )
            loss = regression_loss + domain_loss
            # Calculate gradients
            loss.backward()
            # Update Weights
            self.optimizer.step()
            # Calculate Loss
            regression_training_loss += regression_loss.item()
            domain_training_loss += domain_loss.item()

            number_of_images += len(images)
            i += 1

        return (
            regression_training_loss / number_of_images,
            domain_training_loss / number_of_images,
        )

    def compute_validation_loss(self):
        """
        Compute the validation loss for the current epoch

        Returns:
            float: The validation loss
        """
        with torch.no_grad():
            self.model.eval()

            domain_validation_loss, regression_validation_loss = 0.0, 0.0
            number_of_images = 0
            len_dataloader = len(self.validation_loader)
            i = 0
            for images, labels, domains in tqdm(
                self.validation_loader, "validation", leave=False
            ):
                # https://github.com/fungtion/DANN
                p = (
                    float(i + self.epoch * len_dataloader)
                    / self.NB_EPOCHS
                    / len_dataloader
                )
                # https://github.com/fungtion/DANN
                alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
                images, labels, domains = (
                    images.to(self.device),
                    labels.to(self.device),
                    domains.to(self.device),
                )

                # Forward Pass
                predictions, classifications = self.model(images, 2 * alpha)
                # Find the Loss
                regression_loss = self.loss_function(predictions, labels)
                domain_loss = self.domain_classification_loss_function(
                    classifications, domains.unsqueeze(1)
                )
                # Calculate Loss
                regression_validation_loss += regression_loss.item()
                domain_validation_loss += domain_loss.item()
                number_of_images += len(images)
                i += 1

            return (
                regression_validation_loss / number_of_images,
                domain_validation_loss / number_of_images,
            )

    def update_plot(self):
        """
        Updates the plot at the end of an epoch to show all of the training
        losses and validation losses computed so far
        """
        self.ax1.clear()
        self.ax2.clear()

        self.ax1.plot(
            range(len(self.training_regression_losses)),
            self.training_regression_losses,
            label="training regression loss",
        )
        self.ax1.plot(
            range(len(self.validation_regression_losses)),
            self.validation_regression_losses,
            label="validation regression loss",
        )

        self.ax2.plot(
            range(len(self.training_regression_losses)),
            self.training_domain_losses,
            label="training domain loss",
        )
        self.ax2.plot(
            range(len(self.validation_regression_losses)),
            self.validation_domain_losses,
            label="validation domain loss",
        )
        self.ax1.legend(loc="upper left")
        self.ax2.legend(loc="upper left")

        self.figure.canvas.draw()
        plt.pause(0.001)
