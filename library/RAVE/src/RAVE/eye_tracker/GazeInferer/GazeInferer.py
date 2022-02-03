import torch
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from RAVE.eye_tracker.GazeInferer.deepvog.eyefitter import SingleEyeFitter

"""
This file is a combination of multiple files of deepvog. It regroups the elements that are specific to our use case only, and 
uses pytorch instead of TF for predictions. As such, this is mostly copy and pasted (and adapted) code from deepvog.   
"""

class GazeInferer:
    def __init__(self, ellipse_dnn, dataloader, device, eyeball_model_path = "model_01.json", image_scaling_factor=1,  pupil_radius=4, initial_eye_z=8, flen=3.37, sensor_size=(2.7216, 3.6288)):
        self._ellipse_dnn = ellipse_dnn 
        self._dataloader = dataloader
        self._device = device
        self._eyeball_model_path = eyeball_model_path

        image, _, _ = next(iter(self._dataloader))  
        self.shape = image.shape[2], image.shape[3]
        self.mm2px_scaling = np.linalg.norm(self.shape) / np.linalg.norm(sensor_size)

        #TODO FC : deal with image_scaling_factor when we'll have real images
        self._eyefitter = SingleEyeFitter(focal_length=flen * self.mm2px_scaling * image_scaling_factor,
                                    pupil_radius = pupil_radius * self.mm2px_scaling * image_scaling_factor,
                                    initial_eye_z = initial_eye_z * self.mm2px_scaling,
                                    image_shape=self.shape)
        
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
                predictions =  self._ellipse_dnn(images)

                for prediction in predictions:
                    self._eyefitter.unproject_single_observation(self.torch_prediction_to_deepvog_format(prediction))
                    self._eyefitter.add_to_fitting()

        # Fit eyeball models. Parameters are stored as internal attributes of Eyefitter instance.
        self._eyefitter.fit_projected_eye_centre(ransac=True, max_iters=100, min_distance=10*self._dataloader.dataset.__len__())
        self._eyefitter.estimate_eye_sphere()

        # Issue error if eyeball model still does not exist after fitting.
        if (self._eyefitter.eye_centre is None) or (self._eyefitter.aver_eye_radius is None):
            raise TypeError("Eyeball model was not fitted. You may need -v or -m argument to check whether the pupil segmentation works properly.")
        
        self.save_eyeball_model()
    
    def torch_prediction_to_deepvog_format(self, prediction):
            HEIGHT, WIDTH = self.shape[0], self.shape[1]
            prediction = prediction.cpu().numpy()
            cx, cy, w, h, radian = prediction[0], prediction[1], prediction[2], prediction[3], prediction[4]
            cx, cy, w, h, radian = WIDTH*cx, HEIGHT*cy, WIDTH*w, HEIGHT*h, 2*torch.pi*radian
            return [cx, cy], w, h, radian

    def save_eyeball_model(self):
        save_dict = {"eye_centre": self._eyefitter.eye_centre.tolist(), "aver_eye_radius": self._eyefitter.aver_eye_radius}
        json_str = json.dumps(save_dict, indent=4)
        with open(self._eyeball_model_path, "w") as fh:
            fh.write(json_str)


    def infer(self):
        self.load_eyeball_model()

        with torch.no_grad():
            for images, labels, gts in self._dataloader:
                images = images.to(self._device)
                labels = labels.to(self._device)

                predictions =  self._ellipse_dnn(images)

                for prediction, gt in zip(predictions, gts): 
                    self._eyefitter.unproject_single_observation(self.torch_prediction_to_deepvog_format(prediction))
                    _, n_list, _, _ = self._eyefitter.gen_consistent_pupil()
                    x, y = self._eyefitter.convert_vec2angle31(n_list[0])
                    x -= 90
                    y -= 90
                    
                    self.gts_x.append(gt[0])
                    self.gts_y.append(gt[1])

                    self.preds_x.append(x)
                    self.preds_y.append(y)
        
        self.gts_x = np.array( self.gts_x)
        self.gts_y = np.array( self.gts_y) 



        past_x_predictions = []
        past_y_predictions = []
        NB_ROLLING_ELEMENTS = 2
        for i in range(len(self.preds_x)):
            current_prediction = self.preds_x[i]
            past_x_predictions.append(current_prediction)
            if(len(past_x_predictions) > NB_ROLLING_ELEMENTS):
                past_x_predictions.pop(0)
                current_prediction = 0
                for past_prediction in past_x_predictions : 
                    current_prediction += (past_prediction/NB_ROLLING_ELEMENTS)

            self.preds_x[i] = current_prediction

        for i in range(len(self.preds_y)):
            current_prediction = self.preds_y[i]
            past_y_predictions.append(current_prediction)
            if(len(past_y_predictions) > NB_ROLLING_ELEMENTS):
                past_y_predictions.pop(0)
                current_prediction = 0
                for past_prediction in past_y_predictions : 
                    current_prediction += (past_prediction/NB_ROLLING_ELEMENTS)

            self.preds_y[i] = current_prediction

        self.preds_x = np.array( self.preds_x) 
        self.preds_y = np.array( self.preds_y) 

        x_error, y_error = 0, 0
        for i in range(len(self.preds_x)):
            x_error += np.abs(self.gts_x[i] - self.preds_x[i])
            y_error += np.abs(self.gts_y[i] - self.preds_y[i])

        print("mean_x_error = {} ; mean_y_error = {} ".format(x_error/len(self.preds_x), y_error/len(self.preds_x)))
        print("max_x_error = {} ; max_y_error = {} ".format((np.abs(self.gts_x - self.preds_x)).max(), np.abs((self.gts_y - self.preds_y)).max()) )

        plt.figure(0)
        plt.plot( range(len(self.gts_x)), self.gts_x )
        plt.plot( range(len(self.preds_x)), self.preds_x )

        plt.figure(1)
        plt.plot( range(len(self.gts_y)), self.gts_y )
        plt.plot( range(len(self.preds_y)), self.preds_y )
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