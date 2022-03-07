import socketio
import os
from datetime import datetime

sio = socketio.Client()


def emit(eventName, destination, payload):
    sio.emit(eventName, {"destination": destination, "payload": payload})


@sio.event
def connect():
    print("connection established to server")
    # Emit the socket id to the server to "authenticate yourself"
    emit("pythonSocketAuth", "server", {"socketId": sio.get_sid()})


class EyeTrackerCalib:
    def __init__(self):
        self.list_calibs = []
        dir_list = os.listdir("eyetracking_calibrations")
        for file_name in dir_list:
            self.list_calibs.append({"name": file_name.rstrip(".json")})
        self.activate = False
        self.selection = None
        emit("configList", "client", {"configuration": self.list_calibs})
        sio.on("addEyeTrackingCalib", self.add_new_config)
        sio.on("deleteEyeTrackingCalib", self.delete_config)
        sio.on("selectEyeTrackingCalib", self.select_config)
        sio.on("goToEyeTrackingCalibration", self.activate_on)
        sio.on("startEyeTrackingCalibration", self.start_calib)
        sio.on("nextCalibStep", self.next_step)

    def activate_on(self):
        self.activate = True
        emit("configList", "client", {"configuration": self.list_calibs})

    def add_new_config(self, new_name):
        self.list_calibs.append(
            {
                "name": new_name["configName"]
                + datetime.now().strftime("-%d-%m-%Y %H:%M:%S")
            }
        )
        emit("configList", "client", {"configuration": self.list_calibs})

    def delete_config(self, config_name):
        self.list_calibs[:] = [
            d for d in self.list_calibs if d.get("name") != config_name["id"]
        ]
        emit("configList", "client", {"configuration": self.list_calibs})

    def select_config(self, selection_name):
        self.selection = [
            d["name"] + ".json"
            for d in self.list_calibs
            if selection_name["name"] == d.get("name")
        ]
        print(self.selection)

    def start_calib(self):
        print("Start")

    def next_step(self):
        print("Next")


@sio.event
def disconnect():
    print("disconnected from server")


if __name__ == "__main__":
    sio.connect("ws://localhost:9000")
    eyeTrackerCalib = EyeTrackerCalib()
