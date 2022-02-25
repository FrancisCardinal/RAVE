import { AbstractMessage, MESSAGE_DESTINATIONS } from "./abstractMessages";
interface AbstractServerMessage extends AbstractMessage {
  event: SERVER_EVENTS;
  destination : MESSAGE_DESTINATIONS.SERVER;
}

export enum SERVER_EVENTS {
  PYTHON_SOCKET_AUTH = 'pythonSocketAuth',
}

interface PythonSocketPayload {
  socketId : number;
}

export function PythonSocketEvent(socketId : number) {
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: SERVER_EVENTS.PYTHON_SOCKET_AUTH,
    payload : {
      socketId
    } as PythonSocketPayload
  }
};