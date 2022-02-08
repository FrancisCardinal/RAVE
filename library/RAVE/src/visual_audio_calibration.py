import pyodas.visualize as vis
import socketio
from pyodas.io import MicSource
import base64
import cv2
import time

# socket io client
sio = socketio.Client()


@sio.event
def connect():
    print("connection established to server")
    # Emit the socket id to the server to "authenticate yourself"
    sio.emit("pythonSocket", sio.get_sid())


class Calib:
    def __init__(self):
        self.start = False
        self.key = -1
        self.nb_points = 5
        self.order_of_polynomial = 3
        self.CHUNK_SIZE = 512
        self.FRAME_SIZE = 2 * self.CHUNK_SIZE
        self.CHANNELS = 1
        self.WIDTH = 640
        self.HEIGHT = 480
        self.update_params = False
        self.calibration = vis.AcousticImageCalibration(
            self.CHANNELS,
            self.FRAME_SIZE,
            self.WIDTH,
            self.HEIGHT,
            nb_of_points=self.nb_points,
            save_path="./visual_calibration.json",
        )
        self.video_source = vis.VideoSource(0, self.WIDTH, self.HEIGHT)
        self.m = vis.Monitor(
            "Camera", (self.WIDTH, self.HEIGHT), refresh_rate=100
        )

        # Core
        self.mic_source = MicSource(self.CHANNELS, chunk_size=self.CHUNK_SIZE)
        sio.on("nextCalibTarget", self.go_next_target)
        sio.on("changeCalibParams", self.change_nb_points)
        sio.on("startCalibration", self.start_calib)
        sio.on("stopCalibration", self.stop_calib)

    # @sio.on("nextCalibTarget")
    def go_next_target(self):
        print("Client togged next target")
        self.key = 32

    def start_calib(self):
        print("Start calibration")
        self.start = True

    def stop_calib(self):
        print("Stop Calibration")
        self.start = False

    def reset_key(self):
        self.key = -1

    def change_nb_points(self, nb):
        self.update_params = True
        try:
            self.nb_points = int(nb["number"])
            self.order_of_polynomial = int(nb["order"])
        except Exception as e:
            sio.emit("calibrationError", str(e))

    def reset_calibration(self):
        self.update_params = False
        try:
            calib.calibration = vis.AcousticImageCalibration(
                calib.CHANNELS,
                calib.FRAME_SIZE,
                calib.WIDTH,
                calib.HEIGHT,
                nb_of_points=calib.nb_points,
                save_path="./visual_calibration.json",
            )
        except Exception as e:
            sio.emit("calibrationError", str(e))

    @staticmethod
    def send_calib_frame(frames):
        frame_string = base64.b64encode(
            cv2.imencode(".jpg", frames)[1]
        ).decode()
        sio.emit(
            "calibFrame",
            {
                "frame": frame_string,
                "dimensions": frames.shape,
            },
        )


@sio.on("forceRefresh")
def onForceRefresh():
    print("Client called forc e refresh, generating new faces")


@sio.on("muteFunction")
def onMuteRequest(request):
    print("Client wants to mute?", request)


@sio.event
def disconnect():
    print("disconnected from server")


if __name__ == "__main__":
    sio.connect("ws://localhost:9000")
    calib = Calib()
    # Visualization

    while calib.m.window_is_alive():
        if calib.update_params:
            calib.reset_calibration()

        # Get the audio signal and image frame
        x = calib.mic_source()
        frame = calib.video_source()

        # If you are using a webcam of a camera facing yourself, this
        # might feel more natural
        # frame = cv2.flip(frame, 1)

        # Draw the targets on the frame and process the audio signal x
        if calib.start:
            frame = calib.calibration(frame, x, calib.key)
            calib.reset_key()

            calib.send_calib_frame(frame)
            time.sleep(0.01)
        calib.m.update("Camera", frame)
