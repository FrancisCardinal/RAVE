import torch
import os
from threading import Thread
from datetime import datetime

from RAVE.eye_tracker.EyeTrackerTrainer import EyeTrackerTrainer

from RAVE.eye_tracker.EyeTrackerModel import EyeTrackerModel
from RAVE.eye_tracker.EyeTrackerDataset import (
    EyeTrackerDataset,
    EyeTrackerInferenceDataset,
)
from RAVE.eye_tracker.GazeInferer.GazeInferer import GazeInferer


class GazeInfererManager:
    """Wrapper class that builds a GazeInferer object and that manages it.
    Notably, it uses thread to ensure that the GazeInferer does not block
    the other modules when it is adding images to the calibration or
    when it is infering. It also creates the various objects needed by
    the GazeInferer object (such as a dataloader). It also manages the
    calibration file directory and its content.
    """

    IDLE_STATE = 0
    CALIBRATION_STATE = 1
    INFERENCE_STATE = 2

    def __init__(self, CAMERA_INDEX, DEVICE, DEBUG):
        """Constructor of the GazeInfererManager class

        Args:
            CAMERA_INDEX (int): Opencv camera index
            DEVICE (string): pytorch device (likely "cpu" or "cuda")
        """
        self.DEVICE = DEVICE
        self.DEBUG = DEBUG

        self.model = EyeTrackerModel()
        self.model.to(self.DEVICE)
        EyeTrackerTrainer.load_best_model(
            self.model,
            EyeTrackerDataset.EYE_TRACKER_DIR_PATH,
            self.DEVICE,
        )
        self._current_state = GazeInfererManager.IDLE_STATE
        self.gaze_inferer = None
        self.eye_tracker_inference_dataset = EyeTrackerInferenceDataset(CAMERA_INDEX)
        self.selected_calibration_path = None
        self.list_calibration = []
        self.list_available_calibrations()

    def list_available_calibrations(self):
        """Returns the list of known calibration files

        Returns:
            list: The list of known calibration files
        """
        dir_list = os.listdir(GazeInferer.CALIBRATION_MEMORY_PATH)
        self.list_calibration = []
        for file_name in dir_list:
            self.list_calibration.append({"name": file_name.rsplit(".")[0]})
        return self.list_calibration

    def delete_calibration(self, filename):
        """Deletes a calibration file

        Args:
            filename (string): The name of the calibration file to delete.
        """
        file_path = os.path.join(
            GazeInferer.CALIBRATION_MEMORY_PATH,
            filename + ".json",
        )
        self.log("File to delete : {}".format(file_path))
        if os.path.exists(file_path):
            self.log("Deleted!")
            os.remove(file_path)
        self.list_available_calibrations()

    def start_calibration_thread(self):
        """Starts a calibration thread"""
        self.log("Start calibration called")
        self.stop_inference()
        self._current_state = GazeInfererManager.CALIBRATION_STATE

        calibration_loader = torch.utils.data.DataLoader(
            self.eye_tracker_inference_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        self.gaze_inferer = GazeInferer(self.model, calibration_loader, self.DEVICE, self.DEBUG)

        Thread(target=self._add_to_fit, daemon=True).start()

    def stop_inference(self):
        """Stops the inference thread"""
        self.log("Stop Inference called")
        self._current_state = GazeInfererManager.IDLE_STATE

        if self.gaze_inferer is not None:
            self.gaze_inferer.stop_inference()

    def _add_to_fit(self):
        """Creates the relevant objects then starts the acquisition of input
        images.
        """
        self.gaze_inferer.add_to_fit()

    def start_inference_thread(self):
        """Starts the inference thread"""
        self.log("Start inference")
        if self._current_state == GazeInfererManager.CALIBRATION_STATE:
            self.end_calibration_thread()

        self._current_state = GazeInfererManager.INFERENCE_STATE

        conversation_loader = torch.utils.data.DataLoader(
            self.eye_tracker_inference_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        self.gaze_inferer = GazeInferer(self.model, conversation_loader, self.DEVICE, self.DEBUG)
        Thread(target=self._inference, daemon=True).start()

    def pause_calibration_thread(self):
        """Pauses the calibration thread. This does not end the thread, but
        just pauses it momentarily.
        """
        self.log("Pause calibration called")
        self.gaze_inferer.calibration_is_paused = True

    def resume_calibration_thread(self):
        """Resumes a paused calibration thread."""
        self.log("Resume calibration called")
        self.gaze_inferer.calibration_is_paused = False

    def _inference(self):
        """Creates the relevant objects, then calls GazeInferer.infer()"""
        self.gaze_inferer.infer(self.selected_calibration_path)

    def set_selected_calibration_path(self, file_path):
        """Selects a calibration file, so that we know which one to use when
           the inference begins.

        Args:
            file_path (string): The calibration file's path.
        """
        self.log("Select calibration called")
        self.list_available_calibrations()

        self.selected_calibration_path = [
            d["name"] + ".json" for d in self.list_calibration if file_path == d.get("name")
        ]
        self.selected_calibration_path = self.selected_calibration_path[0]
        self.log("Selected {}".format(self.selected_calibration_path))

    def end_calibration_thread(self):
        """Ends a calibration thread"""
        self.log("End calibration called")
        if self.gaze_inferer is not None:
            self.gaze_inferer.stop_adding_to_fit()
            self.gaze_inferer.fit()
            self._current_state = GazeInfererManager.IDLE_STATE

    def set_offset(self):
        """Calls GazeInferer.set_offset()"""
        if self.gaze_inferer is not None:
            self.log("Set offset called")
            self.gaze_inferer.set_offset()

    def save_new_calibration(self, file_name):
        """Saves a new calibration file

        Args:
            file_name (string): The name of the new calibration file
        """
        if self.gaze_inferer is not None:
            file_name = file_name + datetime.now().strftime("-%d-%m-%Y-%H-%M-%S")
            self.gaze_inferer.save_eyeball_model(file_name)
            self.list_calibration.append({"name": file_name})

    def get_current_gaze(self):
        """Returns the latest prediction

        Returns:
            tuple: Pair of x and y angles
        """
        if (self._current_state is not GazeInfererManager.INFERENCE_STATE) or (self.gaze_inferer is None):
            return None, None

        return self.gaze_inferer.get_current_gaze()

    def end(self):
        """Should be called at the end of the program to free the opencv
        device"""
        self.eye_tracker_inference_dataset.end()

    def log(self, msg):
        """Logs a message in the terminal, used for debug

        Args:
            msg (string): The message to display
        """
        print("[Eye tracker] {}".format(msg))
