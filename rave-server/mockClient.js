const { io } = require("socket.io-client");

const socket = io("ws://localhost:9000");

socket.on("connect", () => {
  console.log("Connected as ", socket.id, socket.connected);
  socket.emit("pythonSocket", socket.id);
});

socket.on("forceRefresh", () => {
  console.log("Client called force refresh, generating new faces");
  const faces = [];
  // Generating between 1 and 5 random faces
  const amountOfFaces = Math.floor(Math.random() * 6) + 1;
  for (let i = 0; i < amountOfFaces; i++) {
    faces.push({
      id: i,
      width: 15,
      height: 15,
      dx: Math.floor(Math.random() * 151),
      dy: Math.floor(Math.random() * 126),
    });
  }
  socket.emit("newFacesAvailable", faces);
});
