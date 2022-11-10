import pyodas.visualize as vis
import base64
import cv2

from threading import Thread


class CalibrationAudioVision:
    def __init__(self, cap, mic_source, emit):
        self.start = False
        self.key = -1
        self.nb_points = 5
        self.order_of_polynomial = 3
        self.CHUNK_SIZE = 256
        self.FRAME_SIZE = 2 * self.CHUNK_SIZE
        self.CHANNELS = 4
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
        self.video_source = cap
        self._emit = emit

        # Core
        self.mic_source = mic_source

    # @sio.on("nextCalibTarget")
    def go_next_target(self):
        print("Client togged next target")
        self.key = 32

    def start_calib(self):
        print("Start calibration")
        self.start = True
        thread = Thread(target=self._run)
        thread.daemon = True
        thread.start()

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
            self._emit("calibrationError", "client", {"message": str(e)})

    def reset_calibration(self):
        self.update_params = False
        try:
            self.calibration = vis.AcousticImageCalibration(
                self.CHANNELS,
                self.FRAME_SIZE,
                self.WIDTH,
                self.HEIGHT,
                nb_of_points=self.nb_points,
                save_path="./visual_calibration.json",
            )
        except Exception as e:
            self._emit("calibrationError", "client", {"message": str(e)})

    def send_calib_frame(self, frames):
        frame_string = base64.b64encode(
            cv2.imencode(".jpg", frames)[1]
        ).decode()
        self._emit(
            "calibrationFrame",
            "client",
            {
                "frame": frame_string,
                "dimensions": frames.shape,
            },
        )

    def _run(self):
        while self.start:
            if self.update_params:
                self.reset_calibration()

            # Get the audio signal and image frame
            x = self.mic_source()

            frame = self.calibration(self.video_source(), x, self.key)
            self.reset_key()

            self.send_calib_frame(frame)
            # time.sleep(0.01)


# if __name__ == "__main__":
#     sio.connect("ws://localhost:9000")
#     calib = Calib()
#     # Visualization
#
#     while calib.m.window_is_alive():
#         if calib.update_params:
#             calib.reset_calibration()
#
#         # Get the audio signal and image frame
#         x = calib.mic_source()
#         frame = calib.video_source()
#
#         # If you are using a webcam of a camera facing yourself, this
#         # might feel more natural
#         # frame = cv2.flip(frame, 1)
#
#         # Draw the targets on the frame and process the audio signal x
#         if calib.start:
#             frame = calib.calibration(frame, x, calib.key)
#             calib.reset_key()
#
#             calib.send_calib_frame(frame)
#             time.sleep(0.01)
#         calib.m.update("Camera", frame)
