import { useEffect, useState, useRef } from 'react';
import { Avatar, Card, CardActions, CardContent, Stack } from '@mui/material';
import io from 'socket.io-client';

function TestRoom() {
  const [socket, setSocket] = useState(null);
  const [faces, setFaces] = useState([]);
  const roomCanvasRef = useRef(null);

  useEffect(() => {
    const ws = io('ws://localhost:9000');
    ws.on('onFacesUpdate', (newFaces) => {
      newFaces.forEach((face) => {
        face.color = '#' + getRandomColor();
      });
      setFaces(newFaces);
    });
    setSocket(ws);
    return () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    if (roomCanvasRef && roomCanvasRef.current) {
      const canvas = roomCanvasRef.current;
      const ctx = canvas.getContext('2d');
      const { width: canvasWidth, height: canvasHeight } = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      ctx.globalAlpha = 1;
      faces.forEach((face) => {
        ctx.beginPath();
        ctx.lineWidth = '1';
        ctx.strokeStyle = face.color;
        ctx.rect(face.dx, face.dy, 15, 15);
        ctx.stroke();
      });
    }
  }, [faces]);

  const getRandomColor = () => Math.floor(Math.random() * 16777215).toString(16);

  return (
    <div className="container mx-auto px-4">
      <h1 className="text-3xl font-bold underline">Room</h1>
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
      <Card className="mt-5" sx={{ minWidth: 275, backgroundColor: 'rgb(51 65 85)' }}>
        <CardContent>
          <h1 className="text-3xl font-bold underline text-white">Faces : </h1>
          <br />
          <Stack direction={'row'} spacing={2}>
            {faces.map((face) => {
              return (
                <Avatar
                  key={face.id}
                  src="invalidImg"
                  sx={{
                    border: '5px',
                    borderStyle: 'solid',
                    borderColor: face.color,
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
              socket.emit('forceRefresh');
            }}
          >
            Force refresh
          </button>
        </CardActions>
      </Card>
    </div>
  );
}

export default TestRoom;
