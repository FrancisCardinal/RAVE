import os
import sys
import cv2

from collections import OrderedDict

env_path = os.path.join(os.path.dirname(__file__), "..")
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Tracker
from pyodas.visualize import VideoSource, Monitor
from pytracking.evaluation.multi_object_wrapper import MultiObjectWrapper

WIDTH = 640
HEIGHT = 480
DEVICE_INDEX = 0

video_source = VideoSource(DEVICE_INDEX, WIDTH, HEIGHT)
video_source.set(cv2.CAP_PROP_FPS, 30)
m = Monitor("Camera", video_source.shape)

tracker_class = Tracker("dimp", "dimp18")
params = tracker_class.get_parameters()

params.tracker_name = tracker_class.name
params.param_name = tracker_class.parameter_name

multiobj_mode = getattr(params, "multiobj_mode", getattr(tracker_class.tracker_class, "multiobj_mode", "default"))

if multiobj_mode == "default":
    tracker = tracker_class.create_tracker(params)
elif multiobj_mode == "parallel":
    tracker = MultiObjectWrapper(tracker_class.tracker_class, params, tracker_class.visdom, fast_load=True)
else:
    raise ValueError("Unknown multi object mode {}".format(multiobj_mode))

id = 1
bbox = [252, 120, 230, 270]  # x, y, w, h
prev_output = OrderedDict()
sequence_object_ids = []
info = OrderedDict()
info["previous_output"] = prev_output
info["init_object_ids"] = [
    id,
]
info["init_bbox"] = OrderedDict({id: bbox})
sequence_object_ids.append(id)

while m.window_is_alive():
    frame = video_source()
    frame_disp = frame.copy()

    if len(sequence_object_ids) > 0:
        info["sequence_object_ids"] = sequence_object_ids
        out = tracker.track(frame, info)
        prev_output = OrderedDict(out)

        if "target_bbox" in out:
            for obj_id, state in out["target_bbox"].items():
                state = [int(s) for s in state]
                cv2.rectangle(
                    frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]), (255, 0, 0), 5
                )

    info = OrderedDict()
    info["previous_output"] = prev_output

    m.update("Camera", frame_disp)
