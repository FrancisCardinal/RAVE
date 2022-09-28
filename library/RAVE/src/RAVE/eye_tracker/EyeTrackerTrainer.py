import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from time import localtime, strftime, time
from datetime import timedelta

from RAVE.common.Trainer import Trainer


class EyeTrackerTrainer(Trainer):
    """Trainer of the Eye tracker module"""

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
        """Constructor of the EyeTrackerTrainer class

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

        total_nb_images, total_visible_pupils = 0, 0
        for _, _, visibility in self.training_loader:
            total_visible_pupils += visibility.sum()
            total_nb_images += visibility.shape[0]

        ratio_of_visible_pupils = total_visible_pupils / total_nb_images
        ratio_of_visible_pupils = ratio_of_visible_pupils.item()
        print("RATIO OF VISIBLE PUPILS =  " + str(ratio_of_visible_pupils))
        self._weights = [1 - ratio_of_visible_pupils, ratio_of_visible_pupils]
        self.pupil_visibility_classification_loss_function = (
            self._weighted_binary_cross_entropy
        )

        self.training_regression_losses, self.validation_regression_losses = (
            [],
            [],
        )
        self.training_visibility_losses, self.validation_visibility_losses = (
            [],
            [],
        )

        self.figure, (self.ax1, self.ax2) = plt.subplots(2, 1, sharey=False)
        plt.ion()
        self.figure.show()

    def _weighted_binary_cross_entropy(self, predictions, targets):
        loss = self._weights[1] * (
            targets * torch.log(predictions + 1e-5)
        ) + self._weights[0] * (
            (1 - targets) * torch.log(1 - predictions + 1e-5)
        )

        return torch.neg(torch.sum(loss))

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
                current_visibility_training_loss,
            ) = self.compute_training_loss()
            (
                current_regression_validation_loss,
                current_visibility_validation_loss,
            ) = self.compute_validation_loss()

            self.training_regression_losses.append(
                current_regression_training_loss
            )
            self.validation_regression_losses.append(
                current_regression_validation_loss
            )

            self.training_visibility_losses.append(
                current_visibility_training_loss
            )
            self.validation_visibility_losses.append(
                current_visibility_validation_loss
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
                    f" : Saved the model"
                    f" Validation visibility loss : "
                    f"{current_visibility_validation_loss:.6f}"
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

        regression_training_loss, visibility_training_loss = 0.0, 0.0
        number_of_images = 0
        for images, labels, visibility in tqdm(
            self.training_loader, "training", leave=False
        ):
            images, labels, visibility = (
                images.to(self.device),
                labels.to(self.device),
                visibility.to(self.device),
            )

            # Clear the gradients
            self.optimizer.zero_grad()
            # Forward Pass
            predictions, visibility_classification = self.model(images)
            # Find the Loss
            # If the pupil is not visible, we do not care what ellipse the
            # network outputs, we do not want to constrain the network in
            # that case, so we zero out the regression gradient if the
            # pupil is not visible (element wise multiplication with
            # visibility_classification)
            predictions = predictions * visibility_classification
            regression_loss = self.loss_function(predictions, labels)

            visibility_loss = (
                3.0
                * self.pupil_visibility_classification_loss_function(
                    visibility_classification, visibility.unsqueeze(1).float()
                )
            )
            loss = regression_loss + visibility_loss
            # Calculate gradients
            loss.backward()
            # Update Weights
            self.optimizer.step()
            # Calculate Loss
            regression_training_loss += regression_loss.item()
            visibility_training_loss += visibility_loss.item()

            number_of_images += len(images)

        return (
            regression_training_loss / number_of_images,
            visibility_training_loss / number_of_images,
        )

    def compute_validation_loss(self):
        """
        Compute the validation loss for the current epoch

        Returns:
            float: The validation loss
        """
        with torch.no_grad():
            self.model.eval()

            regression_validation_loss, visibility_validation_loss = 0.0, 0.0
            number_of_images = 0
            for images, labels, visibility in tqdm(
                self.validation_loader, "validation", leave=False
            ):
                images, labels, visibility = (
                    images.to(self.device),
                    labels.to(self.device),
                    visibility.to(self.device),
                )

                # Clear the gradients
                self.optimizer.zero_grad()
                # Forward Pass
                predictions, visibility_classification = self.model(images)
                # Find the Loss
                # If the pupil is not visible, we do not care what ellipse the
                # network outputs, we do not want to constrain the network in
                # that case, so we zero out the regression gradient if the
                # pupil is not visible (element wise multiplication with
                # visibility_classification)
                predictions = predictions * visibility_classification
                regression_loss = self.loss_function(predictions, labels)

                visibility_loss = (
                    3.0
                    * self.pupil_visibility_classification_loss_function(
                        visibility_classification,
                        visibility.unsqueeze(1).float(),
                    )
                )

                # Calculate Loss
                regression_validation_loss += regression_loss.item()
                visibility_validation_loss += visibility_loss.item()

                number_of_images += len(images)

            return (
                regression_validation_loss / number_of_images,
                visibility_validation_loss / number_of_images,
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
            range(len(self.training_visibility_losses)),
            self.training_visibility_losses,
            label="training visibility loss",
        )
        self.ax2.plot(
            range(len(self.validation_visibility_losses)),
            self.validation_visibility_losses,
            label="validation visibility loss",
        )

        self.ax1.legend(loc="upper left")
        self.ax2.legend(loc="upper left")

        self.figure.canvas.draw()
        plt.pause(0.001)
