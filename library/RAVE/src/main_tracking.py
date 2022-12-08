import argparse
from pyodas.visualize import VideoSource
from RAVE.face_detection.TrackingManager import TrackingManager
from RAVE.common.jetson_utils import is_jetson, process_video_source

CAMERA_DATA = """<?xml version='1.0'?><opencv_storage><cameraMatrix type_id='opencv-matrix'><rows>3</rows>
<cols>3</cols><dt>f</dt><data>347.33973783 0.0 324.87219961 0.0 345.76160498 243.98890458 0.0 0.0 1.0</data>
</cameraMatrix><distCoeffs type_id='opencv-matrix'><rows>5</rows><cols>1</cols><dt>f</dt>
<data>-3.12182472e-01 9.71248263e-02 -4.70897871e-05 1.60933679e-04 -1.31438715e-02</data>
</distCoeffs></opencv_storage>"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face tracking")
    parser.add_argument(
        "--video_source",
        dest="video_source",
        type=str,
        help="Video input source identifier",
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
        default=640,
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
        help="If true, will hide all debug windows",
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
    args = parser.parse_args()

    cap = VideoSource(process_video_source(args.video_source), args.width, args.height)
    frequency = args.freq
    tracking_manager = TrackingManager(
        cap=cap,
        tracker_type="kcf",
        detector_type="yolo",
        verifier_type="resnet_face_18",  # "resnet_face_18",
        frequency=frequency,
        intersection_threshold=-0.2,
        verifier_threshold=0.5,  # 0.25 for resnet
        visualize=not args.headless,
        debug_preprocess=args.show_preprocess,
        debug_detector=args.show_detector,
    )
    tracking_manager.start(args)
