import torch
import os
from threading import Thread
from datetime import datetime

from RAVE.common.DANNTrainer import DANNTrainer

from RAVE.eye_tracker.EyeTrackerModel import EyeTrackerModel
from RAVE.eye_tracker.EyeTrackerDataset import (
    EyeTrackerDataset,
    EyeTrackerInferenceDataset,
)
from RAVE.eye_tracker.GazeInferer.GazeInferer import GazeInferer


class GazeInfererManager:
    IDLE_STATE = 0
    CALIBRATION_STATE = 1
    INFERENCE_STATE = 2

    def __init__(self, CAMERA_INDEX, DEVICE) -> None:
        self.DEVICE = DEVICE

        self.model = EyeTrackerModel()
        self.model.to(self.DEVICE)
        DANNTrainer.load_best_model(
            self.model,
            EyeTrackerDataset.EYE_TRACKER_DIR_PATH,
            self.DEVICE,
        )
        self._current_state = GazeInfererManager.IDLE_STATE
        self.gaze_inferer = None
        self.eye_tracker_inference_dataset = EyeTrackerInferenceDataset(
            CAMERA_INDEX
        )
        self.selected_calibration_path = []
        self.list_calibration = []
        dir_list = os.listdir("RAVE/eye_tracker/GazeInferer/CalibrationMemory")
        for file_name in dir_list:
            self.list_calibration.append({"name": file_name.rstrip(".json")})

    def delete_calibration(self, filename):
        self.list_calibration[:] = [
            d for d in self.list_calibration if d.get("name") != filename
        ]

    def start_calibration_thread(self):
        print("Start calib")
        self.stop_inference()
        self._current_state = GazeInfererManager.CALIBRATION_STATE

        Thread(target=self._add_to_fit, daemon=True).start()

    def stop_inference(self):
        print("Stop Inf")
        self._current_state = GazeInfererManager.IDLE_STATE

        if self.gaze_inferer is not None:
            self.gaze_inferer.stop_inference()

    def _add_to_fit(self):

        calibration_loader = torch.utils.data.DataLoader(
            self.eye_tracker_inference_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        self.gaze_inferer = GazeInferer(
            self.model, calibration_loader, self.DEVICE
        )
        self.gaze_inferer.add_to_fit()

    def start_inference_thread(self):
        Thread(target=self._inference, daemon=True).start()

    def pause_calibration_thread(self):
        print("pause calib")
        pass  # TODO FC

    def resume_calibration_thread(self):
        print("resume calib")
        pass  # TODO FC

    def _inference(self):
        if self._current_state == GazeInfererManager.CALIBRATION_STATE:
            self._end_calibration_thread()

        self._current_state = GazeInfererManager.INFERENCE_STATE

        conversation_loader = torch.utils.data.DataLoader(
            self.eye_tracker_inference_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        self.gaze_inferer = GazeInferer(
            self.model, conversation_loader, self.DEVICE
        )
        self.gaze_inferer.infer(self.selected_calibration_path)

    def set_selected_calibration_path(self, file_path):
        print("Select calib")
        self.selected_calibration_path = [
            d["name"] + ".json"
            for d in self.list_calibration
            if file_path == d.get("name")
        ]
        print(self.selected_calibration_path)

    def end_calibration_thread(self, configName):
        print("end calib")
        if self.gaze_inferer is not None:
            self.gaze_inferer.stop_adding_to_fit()
            self.gaze_inferer.fit(configName)
            self._current_state = GazeInfererManager.IDLE_STATE
            self.list_calibration.append(
                {
                    "name": configName
                    + datetime.now().strftime("-%d-%m-%Y %H:%M:%S")
                }
            )

    def set_offset(self):
        if self.gaze_inferer is not None:
            self.gaze_inferer.set_offset()

    def get_current_gaze(self):
        if (self._current_state is not GazeInfererManager.INFERENCE_STATE) or (
            self.gaze_inferer is None
        ):
            return None, None

        return self.gaze_inferer.get_current_gaze()

    def end(self):
        self.eye_tracker_inference_dataset.end()
