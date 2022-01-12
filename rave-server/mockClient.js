const { io } = require("socket.io-client");

const socket = io("ws://localhost:9000");

socket.on("connect", () => {
  console.log("Connected as ", socket.id, socket.connected);
  socket.emit("pythonSocket", socket.id);

  setInterval(() => {
    socket.emit("forceRefresh", socket.id);
  }, 5000);
});

socket.on("broadcast", (data) => {
  console.log("Client received : ", data);
});

socket.on("forceRefresh", (data) => {
  console.log("Client called force refresh");
});
