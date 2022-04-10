import os
import torch
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from RAVE.eye_tracker.EyeTrackerDataset import EyeTrackerDataset
from RAVE.eye_tracker.GazeInferer.deepvog.eyefitter import SingleEyeFitter

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
        original_image_size_pre_crop=(480, 640),
    ):
        self._ellipse_dnn = ellipse_dnn
        self._dataloader = dataloader
        self._device = device
        self._eyeball_model_path = os.path.join(
            EyeTrackerDataset.EYE_TRACKER_DIR_PATH,
            "GazeInferer",
            eyeball_model_path,
        )

        image, _, _ = next(iter(self._dataloader))
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

        self.gts_x = []
        self.gts_y = []

        self.preds_x = []
        self.preds_y = []

    def fit(self):
        with torch.no_grad():
            for images, labels, _ in tqdm(self._dataloader, "fitting", leave=False):
                images = images.to(self._device)
                labels = labels.to(self._device)

                # Forward Pass
                predictions, _ =  self._ellipse_dnn(images)

                for prediction in predictions:
                    self._eyefitter.unproject_single_observation(self.torch_prediction_to_deepvog_format(prediction))
                    self._eyefitter.add_to_fitting()

        # Fit eyeball models. Parameters are stored as internal attributes of Eyefitter instance.
        self._eyefitter.fit_projected_eye_centre(
            ransac=False,
            max_iters=5000,
            min_distance=10 * len(self._dataloader.dataset),
        )
        self._eyefitter.estimate_eye_sphere()

        # Issue error if eyeball model still does not exist after fitting.
        if (self._eyefitter.eye_centre is None) or (self._eyefitter.aver_eye_radius is None):
            raise TypeError("Eyeball model was not fitted.")

        self.save_eyeball_model()

    def torch_prediction_to_deepvog_format(self, prediction):
        HEIGHT, WIDTH = self.shape[0], self.shape[1]

        h, k, a, b, theta = prediction
        h, k, a, b, theta = h * WIDTH, k * HEIGHT, a * WIDTH, b * HEIGHT, 2*np.pi*theta - np.pi

        if(theta > np.pi/4) and (theta < 3*np.pi/4):
            theta -= np.pi/2
            a, b = b, a

        elif(theta > 3*np.pi/4):
            theta -= np.pi

        elif(theta < -np.pi/4) and (theta > -3*np.pi/4):
            theta += np.pi/2
            a, b = b, a

        elif (theta < -3*np.pi/4):
            theta += np.pi

        h, k, a, b, theta = h.cpu().numpy(), k.cpu().numpy(), a.cpu().numpy(), b.cpu().numpy(), theta.cpu().numpy()

        return [h, k], a, b, theta

    def save_eyeball_model(self):
        save_dict = {"eye_centre": self._eyefitter.eye_centre.tolist(), "aver_eye_radius": self._eyefitter.aver_eye_radius}
        json_str = json.dumps(save_dict, indent=4)
        with open(self._eyeball_model_path, "w") as fh:
            fh.write(json_str)

    def infer(self):
        self.load_eyeball_model()
        x_offset, y_offset = None, None
        with torch.no_grad():
            for images, labels, gts in self._dataloader:
                images = images.to(self._device)
                labels = labels.to(self._device)

                predictions, _ =  self._ellipse_dnn(images)

                for prediction, gt in zip(predictions, gts):
                    self._eyefitter.unproject_single_observation(self.torch_prediction_to_deepvog_format(prediction))
                    _, n_list, _, _ = self._eyefitter.gen_consistent_pupil()
                    x, y = self._eyefitter.convert_vec2angle31(n_list[0])

                    if(x_offset is None):

                        x_offset = x
                        y_offset = y

                    x -= x_offset
                    y -= y_offset

                    self.gts_x.append(gt[0])
                    self.gts_y.append(gt[1])

                    self.preds_x.append(x)
                    self.preds_y.append(y)

        self.gts_x = np.array(self.gts_x)
        self.gts_y = np.array(self.gts_y)

        self.preds_x = np.array(self.preds_x)
        self.preds_y = np.array(self.preds_y)

        x_error, y_error = 0, 0
        for i in range(len(self.preds_x)):
            x_error += np.abs(self.gts_x[i] - self.preds_x[i])
            y_error += np.abs(self.gts_y[i] - self.preds_y[i])

        print("mean_x_error = {} ; mean_y_error = {} ".format(x_error/len(self.preds_x), y_error/len(self.preds_x)))
        print("max_x_error = {} ; max_y_error = {} ".format((np.abs(self.gts_x - self.preds_x)).max(), np.abs((self.gts_y - self.preds_y)).max()))

        plt.figure(0)
        plt.plot(range(len(self.gts_x)), self.gts_x)
        plt.plot(range(len(self.preds_x)), self.preds_x)

        plt.figure(1)
        plt.plot(range(len(self.gts_y)), self.gts_y)
        plt.plot(range(len(self.preds_y)), self.preds_y)
        plt.show(block=True)

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
