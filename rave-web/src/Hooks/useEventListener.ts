import { useContext, useEffect, useState } from "react";
import SocketContext from '../socketContext';
import { CLIENT_EVENTS } from "rave-protocol";

export type WsFunctionHandler = {
  (...args : any) : Function | void | null;
}


export function useEventListener(event: CLIENT_EVENTS, handler : WsFunctionHandler){
  // Prevents refresh on parent component refresh
  const [handlerFunction,] = useState(() => handler);
  const [cleanUpFunction, setCleanUpFunction] = useState<Function | null>(null);
  const ws = useContext(SocketContext);

  useEffect(()=> {
    if(ws?.connected){
      ws.on(event,(payload) => {
        if(handlerFunction instanceof Function){
          const cleanup = handlerFunction(payload);
          cleanup && cleanup !== cleanUpFunction && setCleanUpFunction(cleanup);
        }
      });
    }
    return () => {
      ws?.removeAllListeners(event);
      cleanUpFunction && cleanUpFunction();
    }
  },[cleanUpFunction, event, handlerFunction, ws]);
}