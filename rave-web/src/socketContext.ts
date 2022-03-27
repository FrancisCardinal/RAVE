import React from 'react';
import { Socket } from "socket.io-client"
import { DefaultEventsMap } from 'socket.io-client/build/typed-events';

/**
 * Alias for basic WebSocket type
 */
export type WebSocketType = Socket<DefaultEventsMap, DefaultEventsMap>;

const SocketContext = React.createContext<WebSocketType|null>(null);

export const SocketProvider = SocketContext.Provider;

export default SocketContext;
