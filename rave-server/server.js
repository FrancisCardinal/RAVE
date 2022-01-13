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
const mostRecentFaces = [
  { id: 1, dx: 25, dy: 25 },
  { id: 2, dx: 50, dy: 50 },
  { id: 3, dx: 30, dy: 100 },
  { id: 4, dx: 150, dy: 125 },
];

// Python script socket
let pythonSocket = undefined;
let pythonSocketId = "";

io.on("connection", (socket) => {
  console.log("New socket connection : ", socket.id);

  socket.emit("onFacesUpdate", mostRecentFaces);
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

  socket.on("forceRefresh", () => {
    console.log(socket.id + " requested a forceRefresh");
    pythonSocket && pythonSocket.emit("forceRefresh");
    socket.emit("onFacesUpdate", mostRecentFaces);
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
