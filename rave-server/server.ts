import express from "express";
import http from "http";
import { Server, Socket} from "socket.io";
import { WS_EVENTS, CLIENT_EVENTS, SERVER_EVENTS, PYTHON_EVENTS, MESSAGE_DESTINATIONS } from 'rave-protocol';
var cors = require("cors");
const app = express();
app.use(cors());
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*",
  },
});


// Wifi connection status to prototype (0=no connection, 1=pending, 2=connected)
let connectionStatus = 0; //start off without connection

// Python script socket
let pythonSocket : Socket | undefined = undefined;
let pythonSocketId = "";

interface AnyObject {
  [key: string]: any;
}

io.on("connection", (socket) => {

  console.log("New socket connection : ", socket.id);

  socket.on("disconnect", (reason) => {
    if (socket.id === pythonSocketId) {
      console.log("Python socket disconnected because ", reason);
      connectionStatus = 0;
      io.emit(CLIENT_EVENTS.CONNECTION_STATUS, {status : connectionStatus});
    } else {
      console.log("Web client socket closed : ", socket.id);
    }
  });

  socket.onAny((event : WS_EVENTS, { destination, payload } : { destination : MESSAGE_DESTINATIONS, payload : AnyObject | undefined} ) => {
    switch(destination){
      case MESSAGE_DESTINATIONS.CLIENT:
        handleClientMessage(event as CLIENT_EVENTS, payload);
        break;
      case MESSAGE_DESTINATIONS.PYTHON:
        handlePythonMessage(event as PYTHON_EVENTS, payload);
        break;
      case MESSAGE_DESTINATIONS.SERVER:
        handleServerMessage(event as SERVER_EVENTS, payload);
        break;
      default:
        console.error(`Unknown event type : ${event} was sent from ${socket.id} to ${destination}`);
        break;
    }
  });

});

server.listen(9000, () => {
  console.log("listening on *:9000");
});

function handlePythonMessage(event : PYTHON_EVENTS, payload : AnyObject | undefined){
  if (!pythonSocket) {
    console.log(
      `A user requested ${event} but the pythonSocket is not connected`
    );
    return;
  }

  if(!Object.values(PYTHON_EVENTS).includes(event)){
    console.error(`Python event of type ${event} is unknown`);
    return;
  }
  if(!payload || Object.keys(payload).length === 0) pythonSocket.emit(event);
  else pythonSocket.emit(event,payload);
}

function handleClientMessage(event : CLIENT_EVENTS, payload : AnyObject | undefined){
  if(!Object.values(CLIENT_EVENTS).includes(event)){
    console.error(`Client event of type ${event} is unknown`);
    return;
  }
  io.emit(event,payload);
}

function handleServerMessage(event : SERVER_EVENTS, payload : AnyObject | undefined){
  if(!Object.values(SERVER_EVENTS).includes(event)){
    console.error(`Server event of type ${event} is unknown`);
    return;
  }

  switch(event){
    case SERVER_EVENTS.PYTHON_SOCKET_AUTH:
      authenticatePythonSocket(payload);
      break;
    case SERVER_EVENTS.GET_PYTHON_CONNECTION_STATUS:
      io.emit(CLIENT_EVENTS.CONNECTION_STATUS, {status : connectionStatus});
      break;
  }
}

function authenticatePythonSocket(payload : AnyObject | undefined){
  if(!payload || payload.socketId === undefined){
    console.error("pythonSocket tried to authenticate without sending it's {socketId}");
    return;
  }

  const {socketId} = payload;

  if (io.sockets.sockets.has(socketId)) {
    pythonSocket = io.sockets.sockets.get(socketId);
    pythonSocketId = socketId;
    console.log("Python socket authentified : ", socketId);
    connectionStatus = 2;
    io.emit(CLIENT_EVENTS.CONNECTION_STATUS, {status : connectionStatus});
  } else {
    console.log("Python socket tried to authenticate itself with an unknown socketId");
  }
}