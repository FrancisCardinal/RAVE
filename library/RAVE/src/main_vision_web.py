import argparse
import os
from time import sleep

from RAVE.AppManager import AppManager
from RAVE.common.jetson_utils import is_jetson

CAMERA_DATA = """<?xml version='1.0'?><opencv_storage><cameraMatrix type_id='opencv-matrix'><rows>3</rows>
<cols>3</cols><dt>f</dt><data>340.60994606 0.0 325.7756748 0.0 341.93970667 242.46219777 0.0 0.0 1.0</data>
</cameraMatrix><distCoeffs type_id='opencv-matrix'><rows>5</rows><cols>1</cols><dt>f</dt>
<data>-3.07926877e-01 9.16280959e-02 9.46074597e-04 3.07906550e-04 -1.17169354e-02</data>
</distCoeffs></opencv_storage>"""

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web interface for face tracking")

    parser.add_argument(
        "--video_source",
        dest="video_source",
        type=str,
        help="Video input source identifier for face tracking camera",
        default="0"
        if not is_jetson()
        else f"""v4l2src device=/dev/video0 ! video/x-raw, format=UYVY, width=640, heigth=480, framerate=30/1
         ! nvvidconv ! video/x-raw(memory:NVMM)
         ! nvvidconv ! video/x-raw, format=BGRx
         ! videoconvert ! video/x-raw, format=BGR
         ! videoconvert ! cameraundistort  settings=\"{CAMERA_DATA}\"
         ! videoconvert ! appsink""",
    )
    parser.add_argument(
        "--eye_video_source",
        dest="eye_video_source",
        type=int,
        help="Video input source identifier for eye tracker camera",
        default=1,
    )
    parser.add_argument(
        "--nb_mic_channels",
        dest="nb_mic_channels",
        type=int,
        help="Set the number of microphone channels",
        default=2,
    )
    parser.add_argument(
        "--flip",
        dest="flip",
        help="Flip display orientation by 180 degrees on horizontal axis",
        action="store_true",
    )
    parser.add_argument(
        "--flip_display_dim",
        dest="flip_display_dim",
        help="If true, will flip window dimensions to (height, width)",
        action="store_true",
    )
    parser.add_argument(
        "--height",
        dest="height",
        type=int,
        help="Height of the image to be captured by the camera",
        default=480,
    )
    parser.add_argument(
        "--width",
        dest="width",
        type=int,
        help="Width of the image to be captured by the camera",
        default=640,
    )
    parser.add_argument(
        "--freq",
        dest="freq",
        type=float,
        help="Update frequency for the face detector (for adaptive scaling)",
        default=0.75,
    )
    parser.add_argument(
        "--headless",
        dest="headless",
        help="If true, will show the different tracking frames",
        action="store_true",
    )
    parser.add_argument(
        "--show_preprocess",
        dest="show_preprocess",
        help="If true, will show the preprocess debug window",
        action="store_true",
    )
    parser.add_argument(
        "--show_detector",
        dest="show_detector",
        help="If true, will show the detector debug window",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        help="Display some debugging information",
        action="store_true",
    )
    args = parser.parse_args()

    manager = AppManager(args)

    try:
        manager.start()
        while True:
            sleep(1)
    except KeyboardInterrupt:
        manager.stop()
        quit()
