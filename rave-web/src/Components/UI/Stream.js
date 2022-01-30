import React from 'react';
import { useEffect, useState, useRef, useContext, useCallback } from 'react';
import { Avatar, Card, CardActions, CardContent, Stack } from '@mui/material';
import SocketContext from '../../socketContext';

function Stream() {
  const ws = useContext(SocketContext);
  const [frame, setFrame] = useState({});
  const roomCanvasRef = useRef(null);

  useEffect(() => {
    if (ws) {
      ws.emit('requestNewFrame');
      ws.on('onFrameUpdate', (newFrame) => {
        newFrame.boundingBoxes?.forEach((box) => {
          box.color = '#' + getRandomColor();
        });
        setFrame(newFrame);
      });
      return () => {
        ws.close();
      };
    }
  }, [ws]);

  const onCanvasClick = useCallback(
    (canvas, e) => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      // Check if the click is in any box
      frame.boundingBoxes?.forEach((box) => {
        if (x >= box.dx && x <= box.dx + box.width && y >= box.dy && y <= box.dy + box.height) {
          console.log('Clicked box # ', box.id);
        } else {
          console.log(`Clicked at (${x},${y})`);
        }
      });
    },
    [frame.boundingBoxes]
  );

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
      frame.boundingBoxes?.forEach((box) => {
        ctx.beginPath();
        ctx.lineWidth = '5';
        ctx.strokeStyle = box.color;
        ctx.rect(box.dx, box.dy, box.width, box.height);
        ctx.stroke();
      });
    }
  }, [frame]);

  useEffect(() => {
    if (roomCanvasRef && roomCanvasRef.current) {
      const canvas = roomCanvasRef.current;
      canvas.addEventListener('click', (e) => {
        onCanvasClick(canvas, e);
      });
    }
  }, [roomCanvasRef, onCanvasClick]);

  const getRandomColor = () => Math.floor(Math.random() * 16777215).toString(16);

  return (
    <div className="container mx-auto px-4">
      <div className="flex justify-center">
        <canvas
          ref={roomCanvasRef}
          style={{
            height: '50vh',
            maxWidth: '75vw',
            backgroundColor: 'grey',
          }}
        ></canvas>
      </div>
      <div className="flex justify-center">
        <Card className="mt-5 w-full" sx={{ minWidth: 275, backgroundColor: 'rgb(51 65 85)' }}>
          <CardContent>
            <h1 className="text-3xl font-bold underline text-white">Faces : </h1>
            <br />
            <Stack direction={'row'} spacing={2}>
              {frame.boundingBoxes?.map((box) => {
                return (
                  <Avatar
                    key={box.id}
                    src="invalidImg"
                    sx={{
                      border: '5px',
                      borderStyle: 'solid',
                      borderColor: box.color,
                    }}
                  />
                );
              })}
            </Stack>
          </CardContent>
          <CardActions>
            <button
              className="px-4 py-2 font-semibold text-sm bg-sky-500 text-white rounded-none shadow-sm"
              onClick={() => {
                ws.emit('forceRefresh');
              }}
            >
              Force refresh
            </button>
          </CardActions>
        </Card>
      </div>
    </div>
  );
}

export default Stream;
