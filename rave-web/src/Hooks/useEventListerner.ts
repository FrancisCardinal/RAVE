import React, { useContext, useEffect, useState } from "react";
import SocketContext from '../socketContext';
import { CLIENT_EVENTS } from "rave-protocol";

export type WsFunctionHandler = {
  (...args : any) : Function | void | null;
}


export function useEventListener(event: CLIENT_EVENTS, handler : WsFunctionHandler){

  const [cleanUpFunction, setCleanUpFunction] = useState<Function | null>(null);
  const ws = useContext(SocketContext);
  
  useEffect(()=> {
    if(ws?.connected){
      ws.on(event,(payload) => {
        const cleanup = handler(payload);
        cleanup && setCleanUpFunction(cleanup);
      });
    }
    return () => {
      ws?.removeAllListeners(event);
      cleanUpFunction && cleanUpFunction();
    }
  },[cleanUpFunction, event, handler, ws])
}