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

  // The python script should send a pythonSocket event right after connect
  // this is a replacement for a full on authentification solution
  socket.on("pythonSocket", (socketId) => {
    if (io.sockets.sockets.has(socketId)) {
      pythonSocket = io.sockets.sockets.get(socketId);
      console.log("Python socket authentified : ", socketId);
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
});

io.on("disconnect", (socket) => {
  if (socket.id === pythonSocketId) {
    console.log("Python socket disconnected");
  } else {
    console.log("Web client socket closed : ", socket.id);
  }
});

server.listen(9000, () => {
  console.log("listening on *:9000");
});
