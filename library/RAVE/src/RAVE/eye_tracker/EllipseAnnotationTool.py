import os
import glob
from pathlib import Path

import torch
import cv2
import numpy as np

import json

from RAVE.eye_tracker.NormalizedEllipse import NormalizedEllipse
from RAVE.eye_tracker.ellipse_util import draw_ellipse_on_image


class EllipseAnnotationTool:
    ANNOTATING_STATE = 0
    ANNOTATION_COMPLETED_STATE = 1
    SKIP_STATE = 2
    PASS_STATE = 3
    QUITTING_STATE = 4

    WORKING_DIR = "real_dataset"
    ANNOTATION_FILE_EXTENSION = '.json'

    def __init__(self, root,
                 videos_directory_path="videos",
                 annotations_directory_path="annotations"):

        self._videos_paths = glob.glob(
            os.path.join(root, self.WORKING_DIR, videos_directory_path, '*'))
        self._annotations_directory_path = os.path.join(
            root, self.WORKING_DIR, annotations_directory_path)

        self._points = []
        self._MIN_NB_POINTS_FOR_FIT = 5
        self._video_feed = None
        self._annotation_file = None
        self._state = EllipseAnnotationTool.ANNOTATING_STATE

        self._gamma = 1

        self._window_name = "Annotation tool"
        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._capture_mouse_event)

    def _capture_mouse_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONUP:
            self._points.append((x, y))

        elif event == cv2.EVENT_MBUTTONUP:
            if len(self._points) >= self._MIN_NB_POINTS_FOR_FIT:
                self._state = EllipseAnnotationTool.ANNOTATION_COMPLETED_STATE
            else:
                self._state = EllipseAnnotationTool.SKIP_STATE

    def annotate(self):
        current_video_index = 0
        nb_videos = len(self._videos_paths)
        while((current_video_index < nb_videos) and (self._state != self.QUITTING_STATE)):
            video_path = self._videos_paths[current_video_index]
            self._video_feed = cv2.VideoCapture(video_path)

            if not self._video_feed.isOpened():
                raise IOError(
                    "Cannot open specified file ({})".format(video_path))

            self._annotate_one_video(Path(video_path).stem)
            current_video_index += 1

    def _annotate_one_video(self, video_name):
        current_frame = 0

        self.annotation_file = os.path.join(
            self._annotations_directory_path, video_name) + self.ANNOTATION_FILE_EXTENSION
        annotations = {}
        if os.path.isfile(self.annotation_file):
            with open(self.annotation_file, 'r') as file:
                annotations = json.load(file)

        annoted_frames = list(annotations.keys())
        for i in range(len(annotations.keys())):
            annoted_frames[i] = int(annoted_frames[i])

        success = True
        while(success and self._state != self.QUITTING_STATE):
            success, frame = self._video_feed.read()
            if success and not (current_frame in annoted_frames):
                self._state = EllipseAnnotationTool.ANNOTATING_STATE
                self._annotate_one_frame(frame, annotations, current_frame)
            current_frame += 1

        with open(self.annotation_file, 'w') as json_file:
            json.dump(annotations, json_file, indent=4)

    def _annotate_one_frame(self, frame, annotations, current_frame):
        HEIGHT, WIDTH = frame.shape[0], frame.shape[1]

        self._display_ellipse = True
        self._points = []
        normalized_ellipse = None
        while (self._state == EllipseAnnotationTool.ANNOTATING_STATE):
            altered_frame = frame.copy()
            altered_frame = self.adjust_gamma(altered_frame)

            for point in self._points:
                cv2.drawMarker(altered_frame, point, color=(0, 0, 255))

            if len(self._points) >= self._MIN_NB_POINTS_FOR_FIT:
                ellipse = cv2.fitEllipse(np.array(self._points))
                center_coordinates, axis, angle = ellipse

                # We may or may not need to adjust 'angle' (see this schema :
                # https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga28b2267d35786f5f890ca167236cbc69)
                x_axis, y_axis = max(axis), min(axis)
                if(axis[0] < axis[1]):
                    angle += 90  # degrees

                normalized_ellipse = NormalizedEllipse.get_from_opencv_ellipse(
                    center_coordinates[0], x_axis, center_coordinates[1], y_axis, angle, WIDTH, HEIGHT)

                if(self._display_ellipse):
                    with torch.no_grad():
                        altered_frame = draw_ellipse_on_image(
                            altered_frame, torch.tensor(normalized_ellipse.to_list(), dtype=float, requires_grad=False), color=(0, 255, 0))

            cv2.imshow(self._window_name, altered_frame)
            key = cv2.waitKey(1)

            self._handle_keyboard_input(key)

        if self._state == EllipseAnnotationTool.ANNOTATION_COMPLETED_STATE:
            annotation = normalized_ellipse.to_list()
            annotations[current_frame] = annotation

        elif self._state == EllipseAnnotationTool.SKIP_STATE:
            annotations[current_frame] = [-1, -1, -1, -1, -1]

        elif self._state == EllipseAnnotationTool.PASS_STATE:
            annotations[current_frame] = [-2, -2, -2, -2, -2]

        with open(self.annotation_file, 'w') as json_file:
            json.dump(annotations, json_file, indent=4)

    def adjust_gamma(self, image):
        # Taken from https://pyimagesearch.com/2015/10/05/opencv-gamma-correction/

        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / self._gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def _handle_keyboard_input(self, key):
        if key == ord('q'):
            self._state = self.QUITTING_STATE

        elif key == ord('s'):
            self._state = self.PASS_STATE

        elif key == ord('t'):
            self._display_ellipse = not self._display_ellipse

        elif key == ord('c'):
            if len(self._points) > 0:
                self._points.pop()

        elif key == ord('g'):
            self._gamma += 0.1

        elif key == ord('h'):
            self._gamma -= 0.1
