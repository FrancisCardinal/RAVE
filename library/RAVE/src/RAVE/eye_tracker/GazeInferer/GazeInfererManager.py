import torch
from threading import Thread

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
        self.CAMERA_INDEX = CAMERA_INDEX
        self.DEVICE = DEVICE

        self.model = EyeTrackerModel()
        self.model.to(self.DEVICE)
        DANNTrainer.load_best_model(
            self.model, EyeTrackerDataset.EYE_TRACKER_DIR_PATH, self.DEVICE,
        )
        self._current_state = GazeInfererManager.IDLE_STATE
        self.gaze_inferer = None

    def start_calibration_thread(self):
        self.stop_inference()
        self._current_state = GazeInfererManager.CALIBRATION_STATE

        Thread(target=self._add_to_fit, daemon=True).start()

    def stop_inference(self):
        self._current_state = GazeInfererManager.IDLE_STATE

        if self.gaze_inferer is not None:
            self.gaze_inferer.stop_inference()

    def _add_to_fit(self):
        eye_tracker_calibration_dataset = EyeTrackerInferenceDataset(
            self.CAMERA_INDEX, True
        )
        calibration_loader = torch.utils.data.DataLoader(
            eye_tracker_calibration_dataset,
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

    def _inference(self):
        if self._current_state == GazeInfererManager.CALIBRATION_STATE:
            self._end_calibration_thread()

        self._current_state = GazeInfererManager.INFERENCE_STATE

        eye_tracker_conversation_dataset = EyeTrackerInferenceDataset(
            self.CAMERA_INDEX, True
        )
        conversation_loader = torch.utils.data.DataLoader(
            eye_tracker_conversation_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        self.gaze_inferer = GazeInferer(
            self.model, conversation_loader, self.DEVICE
        )
        self.gaze_inferer.infer()

    def _end_calibration_thread(self):
        self.gaze_inferer.stop_adding_to_fit()
        self.gaze_inferer.fit()

    def get_current_gaze(self):
        if self._current_state is not GazeInfererManager.INFERENCE_STATE:
            return None, None

        return self.gaze_inferer.get_current_gaze()
