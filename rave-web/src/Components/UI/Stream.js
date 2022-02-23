import React from 'react';
import { useEffect, useState, useRef, useContext } from 'react';
import { BrowserView, MobileView } from 'react-device-detect';
import SocketContext from '../../socketContext';

function Stream() {
  const ws = useContext(SocketContext);
  const [frame, setFrame] = useState({});
  const [imgSource, setImgSource] = useState('');
  const roomCanvasRef = useRef(null);

  useEffect(() => {
    if (ws) {
      ws.emit('requestNewFrame');
      ws.on('onFrameUpdate', (newFrame) => {
        newFrame.boundingBoxes?.forEach((box) => {
          box.color = '#' + getRandomColor();
        });
        setImgSource(newFrame.frame);
        setFrame(newFrame);
      });
      
    }
  }, [ws]);

  const onCanvasClick = (e) => {
    const { target: canvas } = e;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    // Check if the click is in any box
    frame.boundingBoxes?.forEach((box) => {
      if (x >= box.dx && x <= box.dx + box.width && y >= box.dy && y <= box.dy + box.height) {
        console.log('Clicked box # ', box.id);
        ws.emit('targetSelect', box.id);
      }
    });
  };

  useEffect(() => {
    if (roomCanvasRef && roomCanvasRef.current && frame.frame) {
      const canvas = roomCanvasRef.current;
      // Resize canvas
      [canvas.width, canvas.height] = [frame.dimensions[1], frame.dimensions[0]];
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      ctx.globalAlpha = 1;
      frame.boundingBoxes?.forEach((box) => {
        ctx.beginPath();
        ctx.lineWidth = '5';
        ctx.strokeStyle = box.color;
        ctx.rect(box.dx, box.dy, box.width, box.height);
        ctx.stroke();
      });
    }
  }, [frame]);

  const getRandomColor = () => Math.floor(Math.random() * 16777215).toString(16);

  return (
    <div className="container px-2">
      <div className="flex flex-col justify-center">
        <BrowserView className='flex justify-center'>
          <div className="flex relative w-4/6 ">
            <div className=''>
              <img src={'data:image/jpeg;base64,' + imgSource} alt={'loading...'} />
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
          <div className="relative w-fit">
            <img src={'data:image/jpeg;base64,' + imgSource} alt={'loading...'} />
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
