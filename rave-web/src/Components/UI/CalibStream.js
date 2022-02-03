import React, { useContext, useEffect, useState, useRef } from "react";
import SocketContext from "../../socketContext";

function CalibStream() {
  const ws = useContext(SocketContext);
  const [frame, setFrame] = useState({});
  const roomCanvasRef = useRef(null);

  useEffect(() => {
    if (ws) {
      ws.on('calibFrameUpdate', (newFrame) => {
        setFrame(newFrame);
      });
      return () => {
        ws.close();
      };
    }
  }, [ws]);

  useEffect(() => {
    if (roomCanvasRef && roomCanvasRef.current && frame.frame) {
      const canvas = roomCanvasRef.current;
      // Resize canvas
      [canvas.width, canvas.height] = [frame.dimensions[1], frame.dimensions[0]];
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      ctx.globalAlpha = 1;
      var img = new Image();
      img.src = 'data:image/jpeg;base64,' + frame.frame;
      ctx.drawImage(img, 0, 0);
    }
  }, [frame]);

  return(
    <div className="m-2">
      <canvas
          ref={roomCanvasRef}
          style={{
            height: '50vh',
            maxWidth: '75vw',
            backgroundColor: 'grey',
          }}
        ></canvas>
    </div>
  );
}

export default CalibStream;