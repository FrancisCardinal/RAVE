const express = require("express");
var cors = require("cors");
const app = express();
app.use(cors());
const http = require("http");
const server = http.createServer(app);
const { Server } = require("socket.io");
const io = new Server(server, {
  cors: {
    origin: "*",
  },
});

// Faces array
let mostRecentFrame = {};

// Wifi connection status to prototype (0=no connection, 1=pending, 2=connected)
let newStatus = 0; //start off without connection

// Python script socket
let pythonSocket = undefined;
let pythonSocketId = "";

io.on("connection", (socket) => {
  console.log("New socket connection : ", socket.id);

  socket.on("requestNewFrame", () => {
    console.log("a new user requested the first frame");
  });

  socket.on("forceRefresh", () => {
    console.log(socket.id + " requested a forceRefresh");
    if (!pythonSocket) {
      console.log(
        "A user requested a force refresh but the pythonSocket is not connected"
      );
      return;
    }
    pythonSocket && pythonSocket.emit("forceRefresh");
    socket.emit("onFrameUpdate", mostRecentFrame);
  });

  socket.on("nextCalibTarget", () => {
      console.log(socket.id + " next calibration target");
      if (!pythonSocket) {
        console.log(
          "A user requested a to change target but the pythonSocket is not connected"
        );
        return;
      }
      pythonSocket && pythonSocket.emit("onNextCalibTarget");
  });

  socket.on("muteFunction", (muteRequest) => {
    console.log(socket.id + " requested a muteFunction");
    if (!pythonSocket) {
      console.log(
        "A user requested a mute function but the pythonSocket is not connected"
      );
      console.log("Want to mute? ", muteRequest);
      return;
    }
    pythonSocket && pythonSocket.emit("muteFunction", (muteRequest));
  });

  socket.on("activateEyeTracking", (setEyeTrackingMode) => {
    console.log(socket.id + " requested a activateEyeTrackingMode");
    if (!pythonSocket) {
      console.log(
        "A user requested a activate eye tracking but the pythonSocket is not connected"
      );
      console.log("Want to activate eye tracking mode? ", setEyeTrackingMode);
      return;
    }
  });

  socket.on("setVolume", (volume) => {
    console.log(socket.id + " requested a setVolume");
    if (!pythonSocket) {
      console.log(
        "A user requested a setVolume but the pythonSocket is not connected"
      );
      console.log("Want to set the volume to? ", volume);
      return;
    }
  });

  

  // The python script should send a pythonSocket event right after connect
  // this is a replacement for a full on authentification solution
  socket.on("pythonSocket", (socketId) => {
    if (io.sockets.sockets.has(socketId)) {
      pythonSocket = io.sockets.sockets.get(socketId);
      console.log("Python socket authentified : ", socketId);
      newStatus = 2;
      //io.emit('getConnectionStatus', newStatus);
    } else {
      console.log(
        "Python socket tried to authenticate itself with an unknown socketId"
      );
    }
  });

  // Python socket will emit new frames when available or when a force request is emitted
  socket.on("newFrameAvailable", (newFrame) => {
    // Save the most recent faces
    mostRecentFrame = newFrame;
    // Send them to the clients
    io.emit("onFrameUpdate", mostRecentFrame);
  });

  // // Python socket will emit the connection status at first and when the status changes
  // socket.on("setConnectionStatus", (connectionStatus) => {
  //   // Save the status
  //   newStatus = connectionStatus;
  //   // Send the status to the clients
  //   io.emit('getConnectionStatus', newStatus);
  // });

});

io.on("disconnect", (socket) => {
  if (socket.id === pythonSocketId) {
    console.log("Python socket disconnected");
    newStatus = 0;
    // io.emit('getConnectionStatus', newStatus);
  } else {
    console.log("Web client socket closed : ", socket.id);
  }
});

server.listen(9000, () => {
  console.log("listening on *:9000");
});
