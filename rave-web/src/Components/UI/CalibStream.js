import React, { useContext, useEffect, useState } from "react";
import SocketContext from "../../socketContext";

function CalibStream() {
  const ws = useContext(SocketContext);
  const [frame, setFrame] = useState({});

  useEffect(() => {
    if (ws) {
      ws.on('onCalibFrame', (newFrame) => {
        setFrame(newFrame);
      });
    }
  }, [ws]);

  return(
    <div className="m-2">
      <img src={'data:image/jpeg;base64,' + frame.frame} alt="calib video" />
    </div>
  );
}

export default CalibStream;