import os
import torch
import numpy as np
import json
from threading import Lock

from RAVE.eye_tracker.EyeTrackerDataset import (
    EyeTrackerDataset,
    EyeTrackerInferenceDataset,
)
from RAVE.eye_tracker.GazeInferer.deepvog.eyefitter import SingleEyeFitter

from RAVE.common.Filters import box_smooth


class GazeInferer:
    """This class is a combination of multiple files of deepvog. It regroups
    the elements that are specific to our use case only, and uses pytorch
    instead of TF for predictions. As such, this is mostly copy and pasted
    (and adapted) code from deepvog. Its goal is to convert an ellipse
    prediction into a gaze prediction. In order to do this, you must first
    run a calibration, in order to compute some metrics about the user's
    eye, and then you can run the inference.
    """

    CALIBRATION_MEMORY_PATH = os.path.join(
        EyeTrackerDataset.EYE_TRACKER_DIR_PATH,
        "GazeInferer",
        "CalibrationMemory",
    )

    def __init__(
        self,
        ellipse_dnn,
        dataloader,
        device,
        DEBUG,
        pupil_radius=4,
        initial_eye_z=52.271,
        flen=5.2,
        sensor_size=(3.24, 5.76),
        original_image_size_pre_crop=(
            EyeTrackerInferenceDataset.ACQUISITION_HEIGHT,
            EyeTrackerInferenceDataset.ACQUISITION_WIDTH,
        ),
    ):
        """Constructor of the GazeInferer class

        Args:
            ellipse_dnn (pytorch model): The eye tracker's neural network
            dataloader (Dataloader): Pytorch's dataloader that enables us to
                get the images of the camera
            device (string): Pytorch device (i.e, should we use cpu or gpu ?)
            DEBUG (bool): Whether to display the debug feed or not.
            pupil_radius (int, optional): Approximation of the radius of the
                observed pupil. Defaults to 4, as this is the median value
                (pupil size is typically between 2-8 mm, if you know your
                observations will be exclusively in a dark or bright room,
                you can change this accordingly, otherwise we suggest using
                the default (median) value)
            initial_eye_z (float, optional): Distance between the camera and
                the surface of the eye (mm). Defaults to 52.271.
            flen (float, optional): Focal length of the camera (mm). Defaults
                to 5.2.
            sensor_size (tuple, optional): Size (Height, Width) of the camera
                sensor (mm). Defaults to (3.24, 5.76).
            original_image_size_pre_crop (tuple, optional): Original image
                size, before any crop or resize (i.e, the acquisition size).
                Defaults to ( EyeTrackerInferenceDataset.ACQUISITION_HEIGHT,
                EyeTrackerInferenceDataset.ACQUISITION_WIDTH, ).
        """
        self._ellipse_dnn = ellipse_dnn
        self._dataloader = dataloader
        self._device = device
        self.DEBUG = DEBUG

        self._selected_calibration_path = ""
        image = next(iter(self._dataloader))
        self.shape = image.shape[2], image.shape[3]

        self.eyefitter = SingleEyeFitter(
            focal_length=flen,
            pupil_radius=pupil_radius,
            initial_eye_z=initial_eye_z,
            image_shape=self.shape,
            original_image_size_pre_crop=original_image_size_pre_crop,
            sensor_size=sensor_size,
        )
        self.x, self.y = None, None
        self._x_offset, self._y_offset = None, None

        self._median_size, self._box_size = 5, 3
        self._past_xs, self._past_ys = (
            np.zeros((self._median_size + self._box_size - 1)),
            np.zeros((self._median_size + self._box_size - 1)),
        )

        self.calibration_is_paused = False

        self._gaze_lock = Lock()

    def add_to_fit(self):
        """Acquires and add images to an internal array, so that they can be
        used to compute a calibration file when the fit() method will be
        called. Adds images while the self.should_add_to_fit flag is True.
        The loop can also be paused momentarily using the
        self.calibration_is_paused flag.
        """
        self.should_add_to_fit, self.calibration_is_paused = True, False
        print("Adding to fitting")
        with torch.no_grad():
            while self.should_add_to_fit:
                if self.calibration_is_paused:
                    continue
                for images in self._dataloader:
                    images = images.to(self._device)

                    # Forward Pass
                    predictions, predicted_visibilities = self._ellipse_dnn(images)

                    if predicted_visibilities[0].item() > 0.90:
                        for prediction in predictions:
                            self.eyefitter.unproject_single_observation(
                                self.torch_prediction_to_deepvog_format(prediction)
                            )
                            self.eyefitter.add_to_fitting()

    def stop_adding_to_fit(self):
        """Sets the self.should_add_to_fit flag to False so that the add_to_fit
        loop stops. Should be called after a call to add_to_fit and before
        a call to fit().
        """
        self.should_add_to_fit = False

    def fit(self):
        """Fit eyeball model with the acquired images. Parameters are stored as
            internal attributes of self.eyefitter

        Raises:
            TypeError: Raised if the eyeball model could not be fitted (we
            received too few or garbage predictions) TODO FC :Recover from this
        """
        self.eyefitter.fit_projected_eye_centre(
            ransac=True,
            max_iters=5000,
            min_distance=np.inf,
        )
        self.eyefitter.estimate_eye_sphere()

        # Issue error if eyeball model still does not exist after fitting.
        if (self.eyefitter.eye_centre is None) or (self.eyefitter.aver_eye_radius is None):
            raise TypeError("Eyeball model was not fitted.")
        else:
            print("Eyeball model was fitted !")

    def torch_prediction_to_deepvog_format(self, prediction):
        """Takes one output of the neural network and converts it to the
           format that deepvog expects. Notably, the predicted ellipse is
           denormalized (i.e, we multiply the predicted values by the
           height and width of the image), the 'a' and 'b' axis may need to be
           switched and a mapping is done between the rotation angle convention
           of the neural network (clockwise angle between the X axis and the
           horizontal axis of the ellipse) and the deepvog convention (which
           we have no clue what it is, we just determined the relationship
           experimentally)

        Args:
            prediction (torch tensor): The prediction

        Returns:
            Tuple: The converted prediction
        """
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
        """The GazeInferer needs to set an x and y offset explicitely.
        There is no way for the system to know what the "rest position"
        of the eye is (i.e, how the eye is when its at (0 deg, 0 deg)).
        This method should be called when the user is looking straight
        ahead (ideally, for more precision, this should be called with
        a protractor that ensures that the user is looking straight
        ahead (horizontally and vertically) (or using trigonometry
        to ensure a physical calibration point is placed at (0,0) ))
        """
        with torch.no_grad():
            while (self._x_offset is None) and (self._y_offset is None):
                for images in self._dataloader:
                    (
                        self._x_offset,
                        self._y_offset,
                    ) = self.get_angles_from_image(images)
                    break

    def save_eyeball_model(self, file_name):
        """Saves the fitted model (and the offset pair) to the disk.
            This file can now be loaded and used to infer the gaze.

        Args:
            file_name (string): name of the calibration file
        """
        full_path = os.path.join(
            self.CALIBRATION_MEMORY_PATH,
            file_name + ".json",
        )
        save_dict = {
            "eye_centre": self.eyefitter.eye_centre.tolist(),
            "aver_eye_radius": self.eyefitter.aver_eye_radius,
            "x_offset": self._x_offset,
            "y_offset": self._y_offset,
        }
        json_str = json.dumps(save_dict, indent=4)
        with open(full_path, "w") as fh:
            fh.write(json_str)

    def infer(self, file_name):
        """Loads a calibration file, then infers the gaze in real time from
           the video feed. Runs while the self.should_infer is set to True

        Args:
            file_name (string): name of the calibration file
        """
        self.load_eyeball_model(file_name)

        self.should_infer = True
        with torch.no_grad():
            while self.should_infer:
                for images in self._dataloader:

                    x, y = self.get_angles_from_image(images)

                    if (x is None) or (y is None):
                        self.x = None
                        self.y = None
                        continue

                    self._past_xs = np.roll(self._past_xs, -1)
                    self._past_ys = np.roll(self._past_ys, -1)
                    self._past_xs[-1] = x
                    self._past_ys[-1] = y

                    box_filtered_x = box_smooth(
                        self._past_xs[self._box_size - 1 : 2 * self._box_size - 1],
                        self._box_size,
                    )
                    box_filtered_y = box_smooth(
                        self._past_ys[self._box_size - 1 : 2 * self._box_size - 1],
                        self._box_size,
                    )

                    self._gaze_lock.acquire()

                    self.x = box_filtered_x[self._box_size // 2] - self._x_offset
                    self.y = box_filtered_y[self._box_size // 2] - self._y_offset

                    self._gaze_lock.release()

        self._gaze_lock.acquire()
        self.x, self.y = None, None
        self._gaze_lock.release()

    def get_angles_from_image(self, images):
        """Takes an image, predicts the ellipse that corresponds to the
           pupil using the neural network, then uses the eye model to
           convert this ellipse to a gave vector (pair of x and y angles)

        Args:
            images (torch tensor): The image on which to run the network
            save_video_feed (bool, optional): Should we save the video feed to
               disk or not ? (used for debug). Defaults to False.

        Returns:
            tuple: pair of x and y angles
        """
        images = images.to(self._device)

        predictions, predicted_visibilities = self._ellipse_dnn(images)
        prediction, visibility = predictions[0], predicted_visibilities[0]

        if visibility.item() < 0.90:
            return None, None

        self.eyefitter.unproject_single_observation(self.torch_prediction_to_deepvog_format(prediction))
        _, n_list, _, _ = self.eyefitter.gen_consistent_pupil()
        x, y = self.eyefitter.convert_vec2angle31(n_list[0])

        return x, y

    def stop_inference(self):
        """Sets the inference flag to False"""
        self.should_infer = False

    def load_eyeball_model(self, file_name):
        """
        Load eyeball model parameters of json format from path.

        Args:
            file_name (string): name of the calibration file
        """
        full_path = os.path.join(self.CALIBRATION_MEMORY_PATH, file_name)
        with open(full_path, "r+") as fh:
            json_str = fh.read()

        loaded_dict = json.loads(json_str)

        self.eyefitter.eye_centre = np.array(loaded_dict["eye_centre"])
        self.eyefitter.aver_eye_radius = loaded_dict["aver_eye_radius"]

        self._x_offset = loaded_dict["x_offset"]
        self._y_offset = loaded_dict["y_offset"]

    def get_current_gaze(self):
        """Returns the latest prediction

        Returns:
            tuple: Pair of x and y angles
        """
        self._gaze_lock.acquire()
        x, y = self.x, self.y  # primitive type, this is a deep copy
        self._gaze_lock.release()

        return x, y
