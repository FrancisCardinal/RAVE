import cv2
import time
import socketio
import base64
import threading

# from tqdm import tqdm

from .face_detection.TrackingManager import TrackingManager
from .face_detection.Pixel2Delay import Pixel2Delay


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


def timed_callback(period, f, *args):
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
    while True:
        time.sleep(next(g))
        f(*args)


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
        self._tracking_manager = TrackingManager(
            tracker_type="kcf",
            detector_type="yolo",
            verifier_type="resnet_face_34",
            frequency=args.freq,
            visualize=args.visualize,
        )
        self._pixel_to_delay = Pixel2Delay(
            (args.height, args.width), "./calibration.json"
        )
        self._args = args
        self._frame_output_frequency = 0.05
        self._delay_update_frequency = 0.25
        self._selected_face = None
        self._vision_mode = "mute"

        sio.on("targetSelect", self._update_selected_face)
        sio.on("changeVisionMode", self._change_mode)

    def start(self):
        """
        Start the tracking loop and the connection to server.
        """
        # Start server
        sio.connect("ws://localhost:9000")

        # Start tracking thread
        threading.Thread(
            target=self._tracking_manager.start,
            args=(self._args,),
            daemon=True,
        ).start()

        # Start the thread that updates the audio target delay
        threading.Thread(
            target=timed_callback,
            args=(self._delay_update_frequency, self._update_target_delays),
            daemon=True,
        ).start()

        # Start the thread that updates the audio target delay
        threading.Thread(
            target=timed_callback,
            args=(self._frame_output_frequency, self.send_to_server),
            daemon=True,
        ).start()

    def send_to_server(self):
        """
        Function to send the frame to the server
        """
        if self._tracking_manager.last_frame is not None:
            boundingBoxes = []
            for obj in self._tracking_manager.tracked_objects.values():
                boundingBoxes.append(
                    {
                        "id": obj.id,
                        "dx": int(obj.bbox[0]),
                        "dy": int(obj.bbox[1]),
                        "width": int(obj.bbox[2]),
                        "height": int(obj.bbox[3]),
                    }
                )

            frame_string = base64.b64encode(
                cv2.imencode(".jpg", self._tracking_manager.last_frame)[1]
            ).decode()
            emit(
                "newFrameAvailable",
                "client",
                {
                    "base64Frame": frame_string,
                    "dimensions": self._tracking_manager.last_frame.shape,
                    "boundingBoxes": boundingBoxes,
                },
            )

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
        if self._selected_face in self._tracking_manager.tracked_objects:
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
