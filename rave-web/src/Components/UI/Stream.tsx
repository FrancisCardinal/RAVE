import React from 'react';
import { useEffect, useState, useRef } from 'react';
import { BrowserView, MobileView } from 'react-device-detect';
import { useEventListener, useEmit } from "../../Hooks";
import { CLIENT_EVENTS, NewFrameAvailablePayload } from 'rave-protocol/clientEvents';
import { TargetSelectEvent } from 'rave-protocol/pythonEvents';

function Stream() {
  const [frame, setFrame] = useState<NewFrameAvailablePayload|null>(null);
  const roomCanvasRef = useRef<HTMLCanvasElement|null>(null);
  const emit = useEmit();

  useEventListener(CLIENT_EVENTS.NEW_FRAME_AVAILABLE,(newFrame : NewFrameAvailablePayload) => {
    newFrame.boundingBoxes.forEach((box) => {
        box.color = '#' + getRandomColor();
    });
    setFrame(newFrame);
  });

  const onCanvasClick = (e : any) => {
    const { target: canvas } = e;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    // Check if the click is in any box
    frame && frame.boundingBoxes.forEach((box) => {
      if (x >= box.dx && x <= box.dx + box.width && y >= box.dy && y <= box.dy + box.height) {
        emit(TargetSelectEvent(box.id));
      }
    });
  };

  useEffect(() => {
    if (roomCanvasRef && roomCanvasRef.current && frame) {
      const canvas = roomCanvasRef.current;
      // Resize canvas
      [canvas.width, canvas.height] = [frame.dimensions[1], frame.dimensions[0]];
      const ctx = canvas.getContext('2d');
      if(ctx){
        ctx && ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        ctx.globalAlpha = 1;
        frame.boundingBoxes?.forEach((box) => {
          ctx.beginPath();
          ctx.lineWidth = 5;
          ctx.strokeStyle = box.color || 'black';
          ctx.rect(box.dx, box.dy, box.width, box.height);
          ctx.stroke();
        });
      }
    }
  }, [frame]);

  const getRandomColor = () => Math.floor(Math.random() * 16777215).toString(16);

  return (
    <div className="container px-2">
      <div className="flex flex-col justify-center">
        <BrowserView className="flex justify-center">
          <div className="flex relative w-4/6 ">
            <div className="select-none">
              {
                frame?.base64Frame ? 
                  <img src={'data:image/jpeg;base64,' + frame?.base64Frame} alt={'loading...'} /> : 
                  <img alt={'loading...'} />
              }
              <canvas
                ref={roomCanvasRef}
                style={{
                  width: '100%',
                  height: '100%',
                  position: 'absolute',
                  top: '0px',
                  left: '0px',
                  backgroundColor: 'rgba(0,0,0,.1)',
                  cursor: 'pointer',
                }}
                onClick={onCanvasClick}
              ></canvas>
            </div>
          </div>
        </BrowserView>
        <MobileView>
          <div className="relative w-fit select-none">
            <img  src={'data:image/jpeg;base64,' + frame?.base64Frame} alt={'loading...'} />
            <canvas
              ref={roomCanvasRef}
              style={{
                width: '100%',
                height: '100%',
                position: 'absolute',
                top: '0px',
                left: '0px',
                backgroundColor: 'rgba(0,0,0,.1)',
                cursor: 'pointer',
              }}
              onClick={onCanvasClick}
            ></canvas>
          </div>
        </MobileView>
      </div>
    </div>
  );
}

export default Stream;
