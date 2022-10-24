import cv2
import time
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

# from RAVE.face_detection.Direction2Pixel import Direction2Pixel

sio = socketio.Client()


@sio.event
def connect():
    """
    Connects the socket to the web.
    """
    print("connection established to server")
    # Emit the socket id to the server to "authenticate yourself"

    emit("pythonSocketAuth", "server", {"socketId": sio.get_sid()})


def emit(event_name, destination, payload):
    """
    Emits event to destination.
    Args:
        event_name (string): The name of the event to emit.
        destination (string):
            The destination to emit the event ("client" or "server").
        payload (dict): The information needed to be passed to the destination.
    """
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
        self._cap.set(cv2.CAP_PROP_FPS, 60)
        self._mic_source = MicSource(4, chunk_size=256)
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
        self._frame_output_frequency = 0.05
        self._delay_update_frequency = 0.25
        self._selected_face = None
        self._vision_mode = "mute"
        self._calibrationAudioVision = CalibrationAudioVision(self._cap, self._mic_source, emit)

        self._gaze_inferer_manager = GazeInfererManager(args.eye_video_source, "cpu")

        sio.on("targetSelect", self._update_selected_face)
        sio.on("changeVisionMode", self._change_mode)
        sio.on("goToEyeTrackingCalibration", self.emit_calibration_list)
        sio.on(
            "startEyeTrackingCalibration",
            self._gaze_inferer_manager.start_calibration_thread,
        )
        sio.on(
            "resumeEyeTrackingCalib",
            self._gaze_inferer_manager.resume_calibration_thread,
        )
        sio.on(
            "pauseEyeTrackingCalib",
            self._gaze_inferer_manager.pause_calibration_thread,
        )
        sio.on(
            "endEyeTrackingCalib",
            self._gaze_inferer_manager.end_calibration_thread,
        )
        sio.on("setOffsetEyeTrackingCalib", self._gaze_inferer_manager.set_offset)
        sio.on("addEyeTrackingCalib", self._save_eye_calibration)
        sio.on("selectEyeTrackingCalib", self._select_eye_tracking_calibration)
        sio.on("deleteEyeTrackingCalib", self._delete_eye_tracking_calibration)
        sio.on("activateEyeTracking", self.control_eye_tracking)

        # Audio-vision calib
        sio.on("nextCalibTarget", self._calibrationAudioVision.go_next_target)
        sio.on("changeCalibParams", self._calibrationAudioVision.change_nb_points)
        sio.on("goToVisionCalibration", self.start_calib_audio_vision)
        sio.on("quitVisionCalibration", self.stop_calib_audio_vision)

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

        # # Anthony: Test to stop all after 10 seconds
        # time.sleep(10)
        # self.stop()

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
        self._selected_face = payload["targetId"]
        emit("selectedTarget", "client", {"targetId": self._selected_face})

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
        self._gaze_inferer_manager.save_new_calibration(payload["configName"])
        self.emit_calibration_list()

    def _select_eye_tracking_calibration(self, payload):
        """
        Assigns the calibration to use in eye-tracking mode.
        Args:
            payload(dict): Containing the calibration filename to use.
        """
        self._gaze_inferer_manager.set_selected_calibration_path(payload["name"])

    def _delete_eye_tracking_calibration(self, payload):
        """
        Deletes the selected eye tracking calibration file.
        """
        self._gaze_inferer_manager.delete_calibration(payload["id"])
        self.emit_calibration_list()

    def control_eye_tracking(self, payload):
        """
        Activate and deactivates the eye tracking control mode.
        Args:
            payload (dict): Key onStatus is a boolean True to
             activate or false to stop
        """
        if payload["onStatus"]:
            self._gaze_inferer_manager.start_inference_thread()
        else:
            self._gaze_inferer_manager.stop_inference()
