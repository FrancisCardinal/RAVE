import cv2
import time
import numpy as np
import socketio
import base64
import threading
from pyodas.visualize import VideoSource
from pyodas.io import MicSource

# from tqdm import tqdm

from .face_detection.TrackingManager import TrackingManager
from .face_detection.Pixel2Delay import Pixel2Delay
from .face_detection.Calibration_audio_vision import CalibrationAudioVision
from .eye_tracker.GazeInferer.GazeInfererManager import GazeInfererManager

from RAVE.face_detection.Direction2Pixel import Direction2Pixel
from RAVE.common.image_utils import bounding_boxes_are_overlapping, box_pair_iou

sio = socketio.Client()


@sio.event
def connect():
    """
    Connects the socket to the web.
    """
    print("connection established to server")
    # Emit the socket id to the server to "authenticate yourself"
    while not sio.connected:
        emit("pythonSocketAuth", "server", {"socketId": sio.get_sid()})
    emit("pythonSocketAuth", "server", {"socketId": sio.get_sid()})


@sio.event
def disconnect():
    """
    Disconnects the socket to the web.
    """
    print("Disconnect to server")


def emit(event_name, destination, payload):
    """
    Emits event to destination.
    Args:
        event_name (string): The name of the event to emit.
        destination (string):
            The destination to emit the event ("client" or "server").
        payload (dict): The information needed to be passed to the destination.
    """
    if sio.connected:
        sio.emit(event_name, {"destination": destination, "payload": payload})


class AppManager:
    """
    Class for managing the output of the tracking to the web

    Args:
        args: Parser args for tracking manager

    Attributes:
        _tracking_manager (TrackingManager):
            Manager for the tracking algorithm
        _pixel_to_delay (Pixel2Delay):
            To obtain the delay of a microphone from a camera pixel
        _args : Parser args for tracking manager
        _frame_output_frequency (flaot):
            Frequency at which we emit a frame to the web
        _delay_update_frequency (float):
            Frequency at which the audio target is updated
        _selected_face (int): Id of the face selected by the web
    """

    def __init__(self, args):
        self._is_alive = True
        self._cap = VideoSource(args.video_source, args.width, args.height)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        self._mic_source = MicSource(args.nb_mic_channels, chunk_size=256)
        self._tracking = True
        self._tracking_manager = TrackingManager(
            cap=self._cap,
            tracker_type="kcf",
            detector_type="yolo",
            verifier_type="arcface",
            frequency=args.freq,
            visualize=args.visualize,
            tracking_or_calib=self.is_tracking,
        )
        self._object_manager = self._tracking_manager.object_manager
        self._pixel_to_delay = Pixel2Delay((args.height, args.width), "./calibration.json")
        self._args = args
        self._frame_output_frequency = 1
        self._delay_update_frequency = 0.25
        self._selected_face = None
        self._vision_mode = "mute"
        self._calibrationAudioVision = CalibrationAudioVision(self._cap, self._mic_source, emit)

        self._gaze_inferer_manager = None
        self._x_1, self._x_2 = None, None

        if args.debug:
            self._tracking_manager.drawing_callbacks.append(self.eye_tracker_debug_drawings)

        sio.on("getTarget", self.get_target)
        sio.on("targetSelect", self._update_selected_face)
        sio.on("changeVisionMode", self._change_mode)

        # Eye-tracker
        sio.on("goToEyeTrackingCalibration", self.emit_calibration_list)
        sio.on(
            "startEyeTrackingCalibration",
            self.start_eye_tracking_calibration,
        )
        sio.on(
            "resumeEyeTrackingCalib",
            self.resume_eye_tracker_calibration_thread,
        )
        sio.on(
            "pauseEyeTrackingCalib",
            self.pause_eye_trackercalibration_thread,
        )
        sio.on(
            "endEyeTrackingCalib",
            self.end_eye_tracking_calibration,
        )
        sio.on("setOffsetEyeTrackingCalib", self.set_eye_tracker_offset)
        sio.on("addEyeTrackingCalib", self._save_eye_calibration)
        sio.on("selectEyeTrackingCalib", self._select_eye_tracking_calibration)
        sio.on("deleteEyeTrackingCalib", self._delete_eye_tracking_calibration)
        sio.on("activateEyeTracking", self.control_eye_tracking)

        # Audio-vision calib
        sio.on("nextCalibTarget", self._calibrationAudioVision.go_next_target)
        sio.on("changeCalibParams", self._calibrationAudioVision.change_nb_points)
        sio.on("goToVisionCalibration", self.start_calib_audio_vision)
        sio.on("quitVisionCalibration", self.stop_calib_audio_vision)

    def _init_eye_tracker(self):
        import torch

        DEVICE = "cpu"
        if torch.cuda.is_available():
            DEVICE = "cuda"

        self._gaze_inferer_manager = GazeInfererManager(self._args.eye_video_source, DEVICE, self._args.debug)

        K = self._tracking_manager.K
        roi = None
        if self._args.undistort:
            K = self._tracking_manager.newcameramtx
            roi = self._tracking_manager.roi

        self._direction_2_pixel = Direction2Pixel(
            K,
            roi,
            np.array([-0.08, 0.05, -0.10]),
            self._args.height,
            self._args.width,
        )

        # Start a thread that selects a face with the GazeInferer if
        # the GazeInferer's inference is running
        threading.Thread(
            target=self.timed_callback,
            args=(
                self._frame_output_frequency,
                self._update_selected_face_from_gaze_inferer,
            ),
            daemon=True,
        ).start()

    def get_target(self):
        emit("selectedTarget", "client", {"targetId": self._selected_face})

    def timed_callback(self, period, f, *args):
        """
        Will call back the function "f" at a timed interval "period"

        Args:
            period (float): Interval in seconds
            f (callable): Function to call back
            *args: Arguments for the function "f"
        """

        def tick():
            """
            Generator to keep the interval constant
            """
            t = time.time()
            while True:
                t += period
                yield max(t - time.time(), 0)

        g = tick()
        while self._is_alive:
            time.sleep(next(g))
            f(*args)

    def emit_calibration_list(self):
        """
        Sends the updated eye tracker calibration list to the server.
        """
        if self._gaze_inferer_manager is None:
            self._init_eye_tracker()
        emit(
            "configList",
            "client",
            {"configuration": self._gaze_inferer_manager.list_calibration},
        )

    def start(self):
        """
        Start the tracking loop and the connection to server.
        """
        # Start server
        sio.connect("ws://localhost:9000")

        # Start tracking thread
        tracking_thread = threading.Thread(
            target=self._tracking_manager.start,
            args=(self._args,),
            daemon=True,
        )
        tracking_thread.start()

        # Start the thread that updates the audio target delay
        target_delays_thread = threading.Thread(
            target=self.timed_callback,
            args=(self._delay_update_frequency, self._update_target_delays),
            daemon=True,
        )
        target_delays_thread.start()

        # Start the thread that updates the audio target delay
        send_to_server_thread = threading.Thread(
            target=self.timed_callback,
            args=(self._frame_output_frequency, self.send_to_server),
            daemon=True,
        )
        send_to_server_thread.start()

    def stop(self):
        """
        Stop the tracking loop and the connection to server.
        """
        # Stop tracking manager
        self._tracking_manager.stop()

        # Stop connection
        self._is_alive = False
        time.sleep(0.1)
        sio.disconnect()

    def is_tracking(self):
        return self._tracking

    def send_to_server(self):
        """
        Function to send the frame to the server
        """
        if self._object_manager.get_last_frame() is not None:
            boundingBoxes = []
            for obj in self._object_manager.tracked_objects.values():
                boundingBoxes.append(
                    {
                        "id": obj.id,
                        "dx": int(obj.bbox[0]),
                        "dy": int(obj.bbox[1]),
                        "width": int(obj.bbox[2]),
                        "height": int(obj.bbox[3]),
                    }
                )

            frame_string = base64.b64encode(cv2.imencode(".jpg", self._object_manager.get_last_frame())[1]).decode()
            emit(
                "newFrameAvailable",
                "client",
                {
                    "base64Frame": frame_string,
                    "dimensions": self._object_manager.get_last_frame().shape,
                    "boundingBoxes": boundingBoxes,
                },
            )

    def start_calib_audio_vision(self):
        print("Start calibration")
        self._tracking = False
        self._calibrationAudioVision.start_calib()

    def stop_calib_audio_vision(self):
        print("Stop Calibration")
        self._calibrationAudioVision.stop_calib()
        self._tracking = True

    def _update_selected_face(self, payload):
        """
        Update the current selected face from the web client
        Args:
            payload (dict): Dictionary containing the id of the
            face selected in the web client.
        """
        if self._selected_face == payload["targetId"]:
            # Deselect the current face
            self._selected_face = None
        else:
            self._selected_face = payload["targetId"]
        emit("selectedTarget", "client", {"targetId": self._selected_face})

    def _update_selected_face_from_gaze_inferer(self):
        if self._gaze_inferer_manager is None:
            self._init_eye_tracker()

        angle_x, _ = self._gaze_inferer_manager.get_current_gaze()
        if angle_x is not None:
            x_1_m, _ = self._direction_2_pixel.get_pixel(angle_x, 0, 1)
            x_5_m, _ = self._direction_2_pixel.get_pixel(angle_x, 0, 5)

            self._x_1, self._x_2 = min(x_1_m, x_5_m), max(x_1_m, x_5_m)

            gaze_bbox = [self._x_1, 0, self._x_2 - self._x_1, self._args.height]
            id, best_iou = None, -1

            for obj in self._object_manager.tracked_objects.values():
                if bounding_boxes_are_overlapping(obj.bbox, gaze_bbox):
                    iou = box_pair_iou(obj.bbox, gaze_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        id = obj.id

            if id is not None:
                self._update_selected_face({"targetId": id})

        else:
            self._x_1, self._x_2 = None, None

    def eye_tracker_debug_drawings(self, frame):
        if self._x_1 is not None:
            cv2.line(
                frame,
                (self._x_1, 0),
                (self._x_1, self._args.height),
                color=(255, 0, 0),
                thickness=2,
            )
            cv2.line(
                frame,
                (self._x_2, 0),
                (self._x_2, self._args.height),
                color=(0, 0, 255),
                thickness=2,
            )

    def _change_mode(self, payload):
        """
        Changes the vision mode when no faces are detected.
        Args:
            payload (dict):
                Dictionary containing Mode of the vision module
                when no faces are detected('mute' or 'hear').
        """
        self._vision_mode = payload["mode"]

    def _update_target_delays(self):
        """
        Function to send the audio section the target delay
        """
        if self._selected_face in self._object_manager.tracked_objects:
            pass
            # print(
            #     self._pixel_to_delay.get_delay(
            #         self._tracking_manager.tracked_objects[
            #             self._selected_face
            #         ].landmark
            #     )
            # )
        else:
            pass
            # print(None)

    def stop_tracking(self):
        """
        Clean tracking
        """
        # TODO-JKealey: won't stop the thread maybe have a bool to skip
        #  tracking part and only output frame, maybe use this for the
        #  force refresh
        self._tracking_manager.stop_tracking()

    def _save_eye_calibration(self, payload):
        """
        Calibration is done and saves the nw calibration in a JSON file.
        Args:
            payload(dict): Contains the filename to use
        """
        if self._gaze_inferer_manager is None:
            self._init_eye_tracker()

        self._gaze_inferer_manager.save_new_calibration(payload["configName"])
        self.emit_calibration_list()

    def _select_eye_tracking_calibration(self, payload):
        """
        Assigns the calibration to use in eye-tracking mode.
        Args:
            payload(dict): Containing the calibration filename to use.
        """
        if self._gaze_inferer_manager is None:
            self._init_eye_tracker()

        self._gaze_inferer_manager.set_selected_calibration_path(payload["name"])

    def _delete_eye_tracking_calibration(self, payload):
        """
        Deletes the selected eye tracking calibration file.
        """
        if self._gaze_inferer_manager is None:
            self._init_eye_tracker()

        self._gaze_inferer_manager.delete_calibration(payload["id"])
        self.emit_calibration_list()

    def control_eye_tracking(self, payload):
        """
        Activate and deactivates the eye tracking control mode.
        Args:
            payload (dict): Key onStatus is a boolean True to
             activate or false to stop
        """
        if self._gaze_inferer_manager is None:
            self._init_eye_tracker()

        if payload["onStatus"]:
            self._gaze_inferer_manager.start_inference_thread()
        else:
            self._gaze_inferer_manager.stop_inference()

    def start_eye_tracking_calibration(self):
        if self._gaze_inferer_manager is None:
            self._init_eye_tracker()

        self._tracking = False
        self._gaze_inferer_manager.start_calibration_thread()

    def end_eye_tracking_calibration(self):
        if self._gaze_inferer_manager is None:
            self._init_eye_tracker()

        self._gaze_inferer_manager.end_calibration_thread()
        self._tracking = True

    def set_eye_tracker_offset(self):
        if self._gaze_inferer_manager is None:
            self._init_eye_tracker()
        self._gaze_inferer_manager.set_offset()

    def resume_eye_tracker_calibration_thread(self):
        if self._gaze_inferer_manager is None:
            self._init_eye_tracker()

        self._gaze_inferer_manager.resume_calibration_thread()

    def pause_eye_trackercalibration_thread(self):
        if self._gaze_inferer_manager is None:
            self._init_eye_tracker()

        self._gaze_inferer_manager.pause_calibration_thread()
