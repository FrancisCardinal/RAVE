import os
import torch
import numpy as np
from scipy.ndimage import median_filter
import json

from RAVE.eye_tracker.EyeTrackerDataset import (
    EyeTrackerDataset,
    EyeTrackerInferenceDataset,
)
from RAVE.eye_tracker.GazeInferer.deepvog.eyefitter import SingleEyeFitter

import cv2
from RAVE.common.image_utils import tensor_to_opencv_image, inverse_normalize
from RAVE.eye_tracker.ellipse_util import draw_ellipse_on_image
from RAVE.common.Filters import box_smooth

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
        self._selected_calibration_path = ""
        image = next(iter(self._dataloader))
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
        self._median_size, self._box_size = 5, 3
        self._past_xs, self._past_ys = (
            np.zeros((self._median_size + self._box_size - 1)),
            np.zeros((self._median_size + self._box_size - 1)),
        )
        self.out = None

    def add_to_fit(self):
        self.should_add_to_fit = True
        print("Adding to fitting")
        with torch.no_grad():
            while self.should_add_to_fit:
                for images in self._dataloader:
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

    def fit(self, configName):
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

        self.save_eyeball_model(configName)

    def torch_prediction_to_deepvog_format(self, prediction):
        HEIGHT, WIDTH = self.shape[0], self.shape[1]

        h, k, a, b, theta = prediction
        h, k, a, b, theta = (
            h * WIDTH,
            k * HEIGHT,
            a * WIDTH,
            b * HEIGHT,
            2 * np.pi * theta - np.pi,
        )

        if (theta > np.pi / 4) and (theta < 3 * np.pi / 4):
            theta -= np.pi / 2
            a, b = b, a

        elif theta > 3 * np.pi / 4:
            theta -= np.pi

        elif (theta < -np.pi / 4) and (theta > -3 * np.pi / 4):
            theta += np.pi / 2
            a, b = b, a

        elif theta < -3 * np.pi / 4:
            theta += np.pi

        h, k, a, b, theta = (
            h.cpu().numpy(),
            k.cpu().numpy(),
            a.cpu().numpy(),
            b.cpu().numpy(),
            theta.cpu().numpy(),
        )

        return [h, k], a, b, theta

    def set_offset(self):
        with torch.no_grad():
            for images in self._dataloader:
                self._x_offset, self._y_offset = self.get_angles_from_image(
                    images
                )
                self.save_eyeball_model()
                break

    def save_eyeball_model(self, file_name):
        full_path = os.path.join(
            EyeTrackerDataset.EYE_TRACKER_DIR_PATH,
            "GazeInferer",
            "CalibrationMemory",
            file_name + ".json",
        )
        save_dict = {
            "eye_centre": self._eyefitter.eye_centre.tolist(),
            "aver_eye_radius": self._eyefitter.aver_eye_radius,
            "x_offset": self._x_offset,
            "y_offset": self._y_offset,
        }
        json_str = json.dumps(save_dict, indent=4)
        with open(full_path, "w") as fh:
            fh.write(json_str)

    def infer(self, name):
        self.load_eyeball_model(name)

        self.should_infer = True
        with torch.no_grad():
            while self.should_infer:
                for images in self._dataloader:

                    x, y = self.get_angles_from_image(images)

                    self._past_xs = np.roll(self._past_xs, -1)
                    self._past_ys = np.roll(self._past_ys, -1)
                    self._past_xs[-1] = x
                    self._past_ys[-1] = y

                    median_filtered_x = median_filter(
                        self._past_xs, self._median_size
                    )
                    median_filtered_y = median_filter(
                        self._past_ys, self._median_size
                    )

                    box_filtered_x = box_smooth(
                        median_filtered_x[
                            self._box_size - 1 : 2 * self._box_size - 1
                        ],
                        self._box_size,
                    )
                    box_filtered_y = box_smooth(
                        median_filtered_y[
                            self._box_size - 1 : 2 * self._box_size - 1
                        ],
                        self._box_size,
                    )

                    self.x = (
                        box_filtered_x[self._box_size // 2] - self._x_offset
                    )
                    self.y = (
                        box_filtered_y[self._box_size // 2] - self._y_offset
                    )

    def get_angles_from_image(self, images, save_video_feed=True):
        images = images.to(self._device)

        predictions, _ = self._ellipse_dnn(images)

        prediction = predictions[0]
        self._eyefitter.unproject_single_observation(
            self.torch_prediction_to_deepvog_format(prediction)
        )
        (
            _,
            n_list,
            _,
            _,
        ) = self._eyefitter.gen_consistent_pupil()
        x, y = self._eyefitter.convert_vec2angle31(n_list[0])

        if save_video_feed:
            if self.out is None:
                self.out = cv2.VideoWriter(
                    "eye_tracker.avi",
                    cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                    30,
                    (320, 240),
                )

            image = inverse_normalize(
                images[0],
                EyeTrackerDataset.TRAINING_MEAN,
                EyeTrackerDataset.TRAINING_STD,
            )
            image = tensor_to_opencv_image(image)

            image = draw_ellipse_on_image(
                image, predictions[0], color=(255, 0, 0)
            )
            self.out.write(image)

        return x, y

    def stop_inference(self):
        self.should_infer = False

    def load_eyeball_model(self, name):
        """
        Load eyeball model parameters of json format from path.

        Args:
            path (str): path of the eyeball model file.
        """
        full_path = os.path.join(
            EyeTrackerDataset.EYE_TRACKER_DIR_PATH,
            "GazeInferer",
            "CalibrationMemory",
            name,
        )
        with open(full_path, "r+") as fh:
            json_str = fh.read()

        loaded_dict = json.loads(json_str)

        self._eyefitter.eye_centre = np.array(loaded_dict["eye_centre"])
        self._eyefitter.aver_eye_radius = loaded_dict["aver_eye_radius"]

        self._x_offset = loaded_dict["x_offset"]
        self._y_offset = loaded_dict["y_offset"]

    def get_current_gaze(self):
        return self.x, self.y
