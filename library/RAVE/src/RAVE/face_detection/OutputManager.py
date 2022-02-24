import cv2
import time
import socketio
import base64
import threading

# from tqdm import tqdm

from .TrackingManager import TrackingManager


sio = socketio.Client()


@sio.event
def connect():
    """
    Connects the socket to the web.
    """
    print("connection established to server")
    # Emit the socket id to the server to "authenticate yourself"
    sio.emit("pythonSocket", sio.get_sid())


class OutputManager:
    """
    Class for managing the output of the tracking to the web

    Args:
        args: Parser args for tracking manager

    Attributes:
        _args : Parser args for tracking manager
        _output_frequency (flaot):
            Frequency at which we emit a frame to the web
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
        self._args = args
        self._output_frequency = 0.05
        self._selected_face = None

        sio.on("targetSelect", self._update_selected_face)

    def start(self):
        """
        Start the tracking loop and the connection to server.
        """
        # Start tracking
        new_thread = threading.Thread(
            target=self._tracking_manager.start,
            args=(self._args,),
            daemon=True,
        )
        new_thread.start()
        # Start server
        sio.connect("ws://localhost:9000")
        self._time = time.time()

        self.send_to_server()

    def send_to_server(self):
        """
        Loop to send the frame to the server
        """
        while True:
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
                sio.emit(
                    "newFrameAvailable",
                    {
                        "frame": frame_string,
                        "dimensions": self._tracking_manager.last_frame.shape,
                        "boundingBoxes": boundingBoxes,
                    },
                )
            time.sleep(self._output_frequency)

    def _update_selected_face(self, ident):
        """
        Update the current selected face from the web client
        Args:
            ident (id): Id of the face selected in the web client
        """
        self._selected_id = ident

    def stop_tracking(self):
        """
        Clean tracking
        """
        # TODO-JKealey: won't stop the thread maybe have a bool to skip
        #  tracking part and only output frame, maybe use this for the
        #  force refresh
        self._tracking_manager.stop_tracking()
