import torch
import numpy as np
import random
import time

from RAVE.eye_tracker.GazeInferer.GazeInfererManager import GazeInfererManager
#from RAVE.face_detection.Direction2Pixel import Direction2Pixel

def inference(device):
    gaze_inferer_manager = GazeInfererManager(1, device)
    """
    head_camera = cv2.VideoCapture(4)
    head_camera.set(cv2.CAP_PROP_FPS, 30.0)

    out = cv2.VideoWriter(
        "head_camera.avi",
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        30,
        (640, 480),
    )

    wait_for_enter("start calibration")
    gaze_inferer_manager.start_calibration_thread()

    wait_for_enter("end calibration")
    gaze_inferer_manager._end_calibration_thread()

    wait_for_enter("set offset")
    gaze_inferer_manager.set_offset()

    wait_for_enter("start inference")
    """
    gaze_inferer_manager.start_inference_thread()

    FPS = 30.0
    #direction_2_pixel = Direction2Pixel(-16, 21)
    for i in range(int(60 * FPS)):
        #ret, frame = head_camera.read()
        time.sleep(1 / FPS)
        angle_x, angle_y = gaze_inferer_manager.get_current_gaze()
        x, y = 0, 0
        if angle_x is not None:
            print("angle_x = {} | angle_y = {}".format(angle_x, angle_y))
        """
            x, y = direction_2_pixel.get_pixel(angle_x, angle_y)

        if ret:
            frame = cv2.flip(frame, 0)
            frame = cv2.flip(frame, 1)
            cv2.drawMarker(frame, (x, y), color=(0, 0, 255), thickness=2)

            out.write(frame)

            cv2.imshow("Facial camera", frame)
            cv2.waitKey(1)
	"""

    gaze_inferer_manager.stop_inference()
    gaze_inferer_manager.end()
    #out.release()


def wait_for_enter(msg=""):
    is_waiting_for_enter = True
    while is_waiting_for_enter:
        key = input("Waiting for enter key to {}".format(msg))
        if key == "":
            is_waiting_for_enter = False


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(0)
    random.seed(42)

    DEVICE = "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda"

    inference(DEVICE)
