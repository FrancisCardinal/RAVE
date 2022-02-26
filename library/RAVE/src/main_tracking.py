import argparse
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
        "--flip_display_dim",
        dest="flip_display_dim",
        type=bool,
        help="If true, will flip window dimensions to (width, height)",
        default=False,
    )
    parser.add_argument(
        "--freq",
        dest="freq",
        type=float,
        help="Update frequency for the face detector (for adaptive scaling)",
        default=1,
    )
    parser.add_argument(
        "--visualize",
        dest="visualize",
        help="If true, will show the different tracking frames",
        action="store_true",
    )
    args = parser.parse_args()

    frequency = args.freq
    tracking_manager = TrackingManager(
        tracker_type="kcf",
        detector_type="yolo",
        verifier_type="resnet_face_34",
        frequency=frequency,
    )
    tracking_manager.start(args)
