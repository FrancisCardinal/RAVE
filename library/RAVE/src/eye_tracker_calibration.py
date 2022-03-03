import socketio
import os

sio = socketio.Client()


@sio.event
def connect():
    print("connection established to server")
    # Emit the socket id to the server to "authenticate yourself"
    sio.emit("pythonSocket", sio.get_sid())


class EyeTrackerCalib:
    def __init__(self):
        self.list_calibs = []
        dir_list = os.listdir("eyetracking_calibrations")
        for file_name in dir_list:
            self.list_calibs.append({"name": file_name.rstrip(".json")})
        self.activate = False
        self.selection = None
        sio.emit("configList", self.list_calibs)
        sio.on("addEyeTrackingCalib", self.add_new_config)
        sio.on("deleteEyeTrackingCalib", self.delete_config)
        sio.on("selectEyeTrackingCalib", self.select_config)
        sio.on("activateEyeTrackingCalibration", self.activate_on)
        sio.on("startEyeTrackingCalibration", self.start_calib)
        sio.on("nextCalibStep", self.next_step)

    def activate_on(self):
        self.activate = True
        sio.emit("configList", self.list_calibs)

    def add_new_config(self, new_name):
        self.list_calibs.append({"name": new_name})
        sio.emit("configList", self.list_calibs)

    def delete_config(self, config_name):
        self.list_calibs[:] = [
            d for d in self.list_calibs if d.get("name") != config_name
        ]
        sio.emit("configList", self.list_calibs)

    def select_config(self, selection_name):
        self.selection = [
            d["name"] + ".json"
            for d in self.list_calibs
            if selection_name == d.get("name")
        ][0]
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
