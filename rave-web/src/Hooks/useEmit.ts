import { useContext } from "react";
import SocketContext from '../socketContext';
import { AbstractMessage } from "rave-protocol";


export function useEmit(){
  const ws = useContext(SocketContext);
  if(ws?.connected){
    return (message : AbstractMessage) => {
      ws.emit(message.event as string,{ destination : message.destination, payload : message.payload});
    }
  }
  return (..._args : any[]) => {
    console.log("Emit was called without Websocket being connected");
  };
}