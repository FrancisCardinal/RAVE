import { CLIENT_EVENTS } from "./clientEvents";
import { PYTHON_EVENTS } from "./pythonEvents";
import { SERVER_EVENTS } from "./serverEvents";

export type WS_EVENTS = CLIENT_EVENTS | PYTHON_EVENTS | SERVER_EVENTS;
export interface AbstractMessage {
  event : CLIENT_EVENTS | PYTHON_EVENTS| SERVER_EVENTS;
  destination : MESSAGE_DESTINATIONS;
  payload? : Object;
}

export enum MESSAGE_DESTINATIONS {
  PYTHON = 'python',
  CLIENT = 'client',
  SERVER = 'server',
}