import argparse
from time import sleep

from RAVE.AppManager import AppManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web interface for face tracking")

    parser.add_argument(
        "--video_source",
        dest="video_source",
        type=int,
        help="Video input source identifier for face tracking camera",
        default=0,
    )
    parser.add_argument(
        "--eye_video_source",
        dest="eye_video_source",
        type=int,
        help="Video input source identifier for eye tracker camera",
        default=1,
    )
    parser.add_argument(
        "--flip_display_dim",
        dest="flip_display_dim",
        help="If true, will flip window dimensions to (width, height)",
        action="store_true",
    )
    parser.add_argument(
        "--flip",
        dest="flip",
        help="Flip display orientation by 180 degrees on horizontal axis",
        action="store_true",
    )
    parser.add_argument(
        "--freq",
        dest="freq",
        type=float,
        help="Update frequency for the face detector (for adaptive scaling)",
        default=1,
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
        default=600,
    )
    parser.add_argument(
        "--dont-visualize",
        dest="visualize",
        help="If true, will show the different tracking frames",
        action="store_false",
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
