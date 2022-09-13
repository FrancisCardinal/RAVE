import React, { useContext, useEffect, useState } from "react";
import SocketContext from '../socketContext';
import { CLIENT_EVENTS } from "rave-protocol";

export type WsFunctionHandler = {
  (...args : any) : Function | void | null;
}


export function useEventListener(event: CLIENT_EVENTS, handler : WsFunctionHandler, deps : React.DependencyList = []){
  // Prevents refresh on parent component refresh
  const [handlerFunction,updateHandlerFunction] = useState(() => handler);
  const [currentDeps, setCurrentDeps] = useState(deps);
  const [cleanUpFunction, setCleanUpFunction] = useState<Function | null>(null);
  const ws = useContext(SocketContext);

  // Check all dependencies to see if they match, if they don't, update
  for(let i = 0; i < currentDeps.length; i++){
    if(currentDeps[i] !== deps[i]){
      setCurrentDeps(deps);
      updateHandlerFunction(() => handler);
    }
  }

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