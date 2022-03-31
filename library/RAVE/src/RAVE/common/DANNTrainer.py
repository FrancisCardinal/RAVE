import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from time import localtime, strftime, time
from datetime import timedelta

from RAVE.common.Trainer import Trainer


class DANNTrainer(Trainer):
    def __init__(
        self,
        real_training_loader,
        real_validation_loader,
        synthetic_training_loader,
        synthetic_validation_loader,
        loss_function,
        device,
        model,
        optimizer,
        scheduler,
        ROOT_DIR_PATH,
        CONTINUE_TRAINING,
    ):
        super().__init__(
            real_training_loader,
            real_validation_loader,
            loss_function,
            device,
            model,
            optimizer,
            scheduler,
            ROOT_DIR_PATH,
            CONTINUE_TRAINING,
        )

        self.synthetic_training_loader = synthetic_training_loader
        self.synthetic_validation_loader = synthetic_validation_loader
        self.domain_classification_loss_function = torch.nn.BCELoss(reduction="sum")

        self.training_regression_losses, self.validation_regression_losses = [], []
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
            current_regression_training_loss, current_domain_training_loss = self.compute_training_loss()
            current_regression_validation_loss, current_domain_validation_loss = self.compute_validation_loss()

            self.training_regression_losses.append(current_regression_training_loss)
            self.validation_regression_losses.append(current_regression_validation_loss)

            self.training_domain_losses.append(current_domain_training_loss)
            self.validation_domain_losses.append(current_domain_validation_loss)
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
                    f"{current_regression_validation_loss:.6f}) : Saved the model"
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

        training_regression_loss, training_domain_loss = 0.0, 0.0
        number_of_images = 0

        len_synthetic_dataloader, len_real_dataloader = len(self.synthetic_training_loader), len(self.training_loader)
        data_synthetic_iter = iter(self.synthetic_training_loader)
        data_real_iter = iter(self.training_loader)

        i = 0
        while i < len_synthetic_dataloader:
            # Compute alpha
            p = (float(i + self.epoch * len_synthetic_dataloader) / self.NB_EPOCHS / len_synthetic_dataloader)  # https://github.com/fungtion/DANN
            # https://github.com/fungtion/DANN
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

            # Clear the gradients
            self.optimizer.zero_grad()

            # Get synthetic predictions and loss
            synthetic_regression_loss, synthetic_domain_loss, nb_synthetic_images = self.get_losses_of_one_batch(data_synthetic_iter, alpha)

            # Get real predictions and loss
            if(i % len_real_dataloader == 0):
                data_real_iter = iter(self.training_loader)
            real_regression_loss, real_domain_loss, nb_real_images = self.get_losses_of_one_batch(data_real_iter, alpha)

            # Total loss
            regression_loss = synthetic_regression_loss + real_regression_loss
            domain_loss = synthetic_domain_loss + real_domain_loss
            loss = regression_loss + domain_loss
            # Calculate gradients
            loss.backward()
            # Update Weights
            self.optimizer.step()
            # Calculate Loss
            training_regression_loss += regression_loss.item()
            training_domain_loss += domain_loss.item()

            number_of_images = number_of_images + nb_synthetic_images + nb_real_images
            i += 1

        return training_regression_loss / number_of_images, training_domain_loss / number_of_images

    def get_losses_of_one_batch(self, data_iter, alpha):
        # Get predictions and loss
        data = data_iter.next()
        images, labels, domains = data

        images, labels, domains = (images.to(self.device), labels.to(self.device), domains.to(self.device),)

        # Forward Pass
        predictions, classifications = self.model(images, alpha)
        # Find the Loss
        loss = self.loss_function(predictions, labels)
        domain_loss = self.domain_classification_loss_function(classifications, domains.unsqueeze(1))

        return loss, domain_loss, len(images)

    def compute_validation_loss(self):
        """
        Compute the validation loss for the current epoch

        Returns:
            float: The validation loss
        """
        with torch.no_grad():
            self.model.eval()

            validation_regression_loss, validation_domain_loss = 0.0, 0.0
            number_of_images = 0

            len_dataloader = min(len(self.synthetic_validation_loader), len(self.validation_loader))
            data_synthetic_iter = iter(self.synthetic_validation_loader)
            data_real_iter = iter(self.validation_loader)

            i = 0
            while i < len_dataloader:
                # Compute alpha
                p = (float(i + self.epoch * len_dataloader) / self.NB_EPOCHS / len_dataloader)  # https://github.com/fungtion/DANN
                # https://github.com/fungtion/DANN
                alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

                # Get synthetic predictions and loss
                synthetic_regression_loss, synthetic_domain_loss, nb_synthetic_images = self.get_losses_of_one_batch(data_synthetic_iter, alpha)

                # Get real predictions and loss
                real_regression_loss, real_domain_loss, nb_real_images = self.get_losses_of_one_batch(data_real_iter, alpha)

                # Total loss
                regression_loss = synthetic_regression_loss + real_regression_loss
                domain_loss = synthetic_domain_loss + real_domain_loss

                # Calculate Loss
                validation_regression_loss += regression_loss.item()
                validation_domain_loss += domain_loss.item()
                number_of_images = number_of_images + nb_synthetic_images + nb_real_images
                i += 1

            return validation_regression_loss / number_of_images, validation_domain_loss / number_of_images

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
