import argparse
import cv2
from pyodas.visualize import VideoSource
from RAVE.face_detection.TrackingManager import TrackingManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face tracking")
    parser.add_argument(
        "--video_source",
        dest="video_source",
        type=int,
        help="Video input source identifier",
        default=0,
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
        help="If true, will flip window dimensions to (width, height)",
        action="store_true",
    )
    parser.add_argument(
        "--undistort",
        dest="undistort",
        help="If true, will correct fish-eye distortion from camera according to hardcoded K & D matrices",
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
        default=600,
    )
    parser.add_argument(
        "--freq",
        dest="freq",
        type=float,
        help="Update frequency for the face detector (for adaptive scaling)",
        default=1,
    )
    parser.add_argument(
        "--headless",
        dest="headless",
        help="If true, will show the different tracking frames",
        action="store_true",
    )
    args = parser.parse_args()

    cap = VideoSource(args.video_source, args.width, args.height)
    cap.set(cv2.CAP_PROP_FPS, 60)

    frequency = args.freq
    tracking_manager = TrackingManager(
        cap=cap,
        tracker_type="kcf",
        detector_type="yolo",
        verifier_type="arcface",
        verifier_threshold=0.5,
        frequency=frequency,
        visualize=not args.headless,
    )
    tracking_manager.start(args)
