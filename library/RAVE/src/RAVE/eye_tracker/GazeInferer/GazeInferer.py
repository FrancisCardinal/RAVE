import os
import torch
import numpy as np
import json

from RAVE.eye_tracker.EyeTrackerDataset import (
    EyeTrackerDataset,
    EyeTrackerInferenceDataset,
)
from RAVE.eye_tracker.GazeInferer.deepvog.eyefitter import SingleEyeFitter
from RAVE.eye_tracker.GazeInferer.deepvog.LSqEllipse import LSqEllipse
from RAVE.eye_tracker.ellipse_util import get_points_of_ellipses

"""
This file is a combination of multiple files of deepvog. It regroups the
elements that are specific to our use case only, and uses pytorch instead of TF
for predictions. As such, this is mostly copy and pasted (and adapted) code
from deepvog.
"""


class GazeInferer:
    def __init__(
        self,
        ellipse_dnn,
        dataloader,
        device,
        eyeball_model_path="model_01.json",
        pupil_radius=4,
        initial_eye_z=52.271,
        x_angle=22.5,
        flen=3.37,
        sensor_size=(2.7216, 3.6288),
        original_image_size_pre_crop=(
            EyeTrackerInferenceDataset.ACQUISITION_HEIGHT,
            EyeTrackerInferenceDataset.ACQUISITION_WIDTH,
        ),
    ):
        self._ellipse_dnn = ellipse_dnn
        self._dataloader = dataloader
        self._device = device
        self._eyeball_model_path = os.path.join(
            EyeTrackerDataset.EYE_TRACKER_DIR_PATH,
            "GazeInferer",
            eyeball_model_path,
        )

        image, _ = next(iter(self._dataloader))
        self.shape = image.shape[2], image.shape[3]

        self._eyefitter = SingleEyeFitter(
            focal_length=flen,
            pupil_radius=pupil_radius,
            initial_eye_z=initial_eye_z,
            x_angle=x_angle,
            image_shape=self.shape,
            original_image_size_pre_crop=original_image_size_pre_crop,
            sensor_size=sensor_size,
        )
        self.x, self.y = None, None

    def add_to_fit(self):
        self.should_add_to_fit = True
        print("Adding to fitting")
        with torch.no_grad():
            while self.should_add_to_fit:
                for images, success in self._dataloader:
                    if not success:
                        continue
                    images = images.to(self._device)

                    # Forward Pass
                    predictions, _ = self._ellipse_dnn(images)

                    for prediction in predictions:
                        self._eyefitter.unproject_single_observation(
                            self.torch_prediction_to_deepvog_format(prediction)
                        )
                        self._eyefitter.add_to_fitting()

    def stop_adding_to_fit(self):
        self.should_add_to_fit = False

    def fit(self):
        # Fit eyeball models. Parameters are stored as internal attributes of
        # Eyefitter instance.
        self._eyefitter.fit_projected_eye_centre(
            ransac=False,
            max_iters=5000,
            min_distance=10 * len(self._dataloader.dataset),
        )
        self._eyefitter.estimate_eye_sphere()

        # Issue error if eyeball model still does not exist after fitting.
        if (self._eyefitter.eye_centre is None) or (
            self._eyefitter.aver_eye_radius is None
        ):
            raise TypeError("Eyeball model was not fitted.")

        self.save_eyeball_model()

    def torch_prediction_to_deepvog_format(self, prediction):
        HEIGHT, WIDTH = self.shape[0], self.shape[1]

        h, k, a, b, theta = prediction
        h, k, a, b = h * WIDTH, k * HEIGHT, a * WIDTH, b * HEIGHT
        x, y = get_points_of_ellipses(
            torch.tensor([h, k, a, b, theta]).unsqueeze(0), 360
        )
        x, y = x.squeeze().cpu().numpy(), y.squeeze().cpu().numpy()

        lsq_ellipse = LSqEllipse()
        lsq_ellipse.fit(x, y)
        center, width, height, radians = lsq_ellipse.parameters()

        return center, width, height, radians

    def save_eyeball_model(self):
        save_dict = {
            "eye_centre": self._eyefitter.eye_centre.tolist(),
            "aver_eye_radius": self._eyefitter.aver_eye_radius,
        }
        json_str = json.dumps(save_dict, indent=4)
        with open(self._eyeball_model_path, "w") as fh:
            fh.write(json_str)

    def infer(self):
        self.load_eyeball_model()
        x_offset, y_offset = None, None

        self.should_infer = True
        with torch.no_grad():
            while self.should_infer:
                for images, success in self._dataloader:
                    if not success:
                        continue
                    images = images.to(self._device)

                    predictions, _ = self._ellipse_dnn(images)

                    for prediction in predictions:
                        self._eyefitter.unproject_single_observation(
                            self.torch_prediction_to_deepvog_format(prediction)
                        )
                        (
                            _,
                            n_list,
                            _,
                            _,
                        ) = self._eyefitter.gen_consistent_pupil()
                        self.x, self.y = self._eyefitter.convert_vec2angle31(
                            n_list[0]
                        )

                        if x_offset is None:
                            # TODO FC : The code assumes that the user will
                            #        be looking straight forward on the first
                            #        frame. Might need to had a calibration
                            #        step for this.
                            x_offset = self.x
                            y_offset = self.y

                        self.x -= x_offset
                        self.y -= y_offset

    def stop_inference(self):
        self.should_infer = False

    def load_eyeball_model(self):
        """
        Load eyeball model parameters of json format from path.

        Args:
            path (str): path of the eyeball model file.
        """
        with open(self._eyeball_model_path, "r+") as fh:
            json_str = fh.read()

        loaded_dict = json.loads(json_str)

        self._eyefitter.eye_centre = np.array(loaded_dict["eye_centre"])
        self._eyefitter.aver_eye_radius = loaded_dict["aver_eye_radius"]

    def get_current_gaze(self):
        return self.x, self.y
