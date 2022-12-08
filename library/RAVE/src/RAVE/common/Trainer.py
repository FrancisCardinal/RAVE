import torch
import os
from time import localtime, strftime, time
from datetime import timedelta
import numpy as np
from tqdm import tqdm

from threading import Thread

import matplotlib.pyplot as plt

plt.ion()


class Trainer:
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

    TRAINING_SESSIONS_DIR = "training_sessions"
    MODEL_INFO_FILE_NAME = "saved_model.pth"

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
        MODEL_INFO_FILE_NAME=None,
    ):
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.loss_function = loss_function
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ROOT_DIR_PATH = ROOT_DIR_PATH

        if MODEL_INFO_FILE_NAME is not None:
            Trainer.MODEL_INFO_FILE_NAME = MODEL_INFO_FILE_NAME
        self.MODEL_PATH = os.path.join(ROOT_DIR_PATH, MODEL_INFO_FILE_NAME)
        self.min_validation_loss = np.inf

        if CONTINUE_TRAINING:
            self.load_model_and_training_info()

        self.training_losses = []
        self.validation_losses = []

        self.terminate_training = False
        Thread(target=self.terminate_training_thread, daemon=True).start()

    def terminate_training_thread(self):
        """
        Thread that checks if the user wants to stop training.
        Used to stop training before all the epochs have been executed.
        """
        while not self.terminate_training:
            key = input()
            if key.upper() == "Q":
                self.terminate_training = True

        print("Terminating training at end of epoch.")

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
            current_training_loss = self.compute_training_loss()
            current_validation_loss = self.compute_validation_loss()

            self.training_losses.append(current_training_loss)
            self.validation_losses.append(current_validation_loss)
            self.update_plot()

            epoch_stats = (
                f"Epoch {self.epoch:0>4d} | "
                f"validation_loss={current_validation_loss:.6f} | "
                f"training_loss={current_training_loss:.6f}"
            )

            if self.min_validation_loss > current_validation_loss:
                epoch_stats = epoch_stats + (
                    "  | Min validation loss decreased("
                    f"{self.min_validation_loss:.6f}--->"
                    f"{current_validation_loss:.6f}) : Saved the model"
                )
                self.min_validation_loss = current_validation_loss

                self.save_model_and_training_info()

            print(epoch_stats)
            if self.scheduler:
                self.scheduler.step(current_validation_loss)
            self.epoch += 1

        self.terminate_training = True
        min_training_loss = min(self.training_losses)
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

        training_loss = 0.0
        number_of_images = 0
        for images, labels in tqdm(self.training_loader, "training", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)

            # Clear the gradients
            self.optimizer.zero_grad()
            # Forward Pass
            predictions = self.model(images)
            # Find the Loss
            loss = self.loss_function(predictions, labels)
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
            for images, labels in tqdm(self.validation_loader, "validation", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward Pass
                predictions = self.model(images)
                # Find the Loss
                loss = self.loss_function(predictions, labels)
                # Calculate Loss
                validation_loss += loss.item()
                number_of_images += len(images)

            return validation_loss / (number_of_images * labels.shape[1] * labels.shape[2])

    def update_plot(self):
        """
        Updates the plot at the end of an epoch to show all of the training
        losses and validation losses computed so far
        """
        plt.clf()
        plt.plot(
            range(len(self.training_losses)),
            self.training_losses,
            label="training loss",
        )
        plt.plot(
            range(len(self.validation_losses)),
            self.validation_losses,
            label="validation loss",
        )
        plt.legend(loc="upper left")
        plt.draw()
        # So that the graph does not get the operating system's
        # focus at each plot update
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(0.001)

    def save_model_and_training_info(self):
        """
        Saves a checkpoint, which contains the model weights and the
        necessary information to continue the training at some other time
        """
        save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "min_validation_loss": self.min_validation_loss,
        }
        if self.scheduler:
            save["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(
            save,
            self.MODEL_PATH,
        )

    def load_model_and_training_info(self):
        """
        Loads a checkpoint, which contains the model weights and the
        necessary information to continue the training now
        """

        checkpoint = torch.load(self.MODEL_PATH)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.min_validation_loss = checkpoint["min_validation_loss"]

        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    @staticmethod
    def load_best_model(model, MODEL_DIR_PATH, device):
        """
        Used to get the best version of a model from disk

        Args:
            model (Module): Model on which to update the weights
            device (string): Torch device (most likely 'cpu' or 'cuda')
        """
        checkpoint = torch.load(
            os.path.join(MODEL_DIR_PATH, Trainer.MODEL_INFO_FILE_NAME),
            map_location=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()
