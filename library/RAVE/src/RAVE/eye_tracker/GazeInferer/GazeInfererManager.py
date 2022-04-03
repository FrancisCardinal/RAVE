import os
import torch
from torch2trt import TRTModule
from threading import Thread

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

        self.model = TRTModule()
        self.model.to(self.DEVICE)
        GazeInfererManager.load_best_model(
            self.model, EyeTrackerDataset.EYE_TRACKER_DIR_PATH, self.DEVICE,
        )
        self._current_state = GazeInfererManager.IDLE_STATE
        self.gaze_inferer = None
        self.eye_tracker_inference_dataset = None
        self._loader = None

    def start_calibration_thread(self):
        self.stop_inference()
        self._current_state = GazeInfererManager.CALIBRATION_STATE

        Thread(target=self._add_to_fit, daemon=True).start()

    def stop_inference(self):
        self._current_state = GazeInfererManager.IDLE_STATE

        if self.gaze_inferer is not None:
            self.gaze_inferer.stop_inference()
    
    def end(self):
        self.eye_tracker_inference_dataset.stop()

    def _add_to_fit(self):
        self.get_new_dataloader()

        self.gaze_inferer = GazeInferer(
            self.model, self._loader, self.DEVICE
        )
        self.gaze_inferer.add_to_fit()

    def get_new_dataloader(self):
        if self.eye_tracker_inference_dataset is not None : 
            self.eye_tracker_inference_dataset.stop()
        
        self.eye_tracker_inference_dataset = EyeTrackerInferenceDataset(
            self.CAMERA_INDEX, True
        )

        self._loader = torch.utils.data.DataLoader(
            self.eye_tracker_inference_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

    def start_inference_thread(self):
        Thread(target=self._inference, daemon=True).start()

    def _inference(self):
        if self._current_state == GazeInfererManager.CALIBRATION_STATE:
            self._end_calibration_thread()

        self.get_new_dataloader()
        self._current_state = GazeInfererManager.INFERENCE_STATE

        self.gaze_inferer = GazeInferer(
            self.model, self._loader, self.DEVICE
        )
        self.gaze_inferer.infer()

    def _end_calibration_thread(self):
        if self.gaze_inferer is not None:
            self.gaze_inferer.stop_adding_to_fit()
            self.gaze_inferer.fit()
            self._current_state = GazeInfererManager.IDLE_STATE

    def set_offset(self):
        if self.gaze_inferer is not None:
            self.gaze_inferer.set_offset()

    def get_current_gaze(self):
        if (self._current_state is not GazeInfererManager.INFERENCE_STATE) or (
            self.gaze_inferer is None
        ):
            return None, None

        return self.gaze_inferer.get_current_gaze()

    @staticmethod
    def load_best_model(model, MODEL_DIR_PATH, device):
        """
        Used to get the best version of a model from disk

        Args:
            model (Module): Model on which to update the weights
        """
        checkpoint = torch.load(
            os.path.join(MODEL_DIR_PATH, "eye_tracker_trt.pth"),
            map_location=device,
        )
        model.load_state_dict(checkpoint)

        model.eval()
