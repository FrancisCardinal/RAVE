import { useEffect, useState } from "react";
import io from "socket.io-client";

function TestRoom() {
  const [socket, setSocket] = useState(null);
  const [faces, setFaces] = useState([]);
  useEffect(() => {
    const ws = io("ws://localhost:9000");
    ws.on("OnFacesUpdate", (newFaces) => {
      setFaces([newFaces]);
    });
    setSocket(ws);
    return () => {
      ws.close();
    };
  }, []);

  return (
    <>
      <h1 className="text-3xl font-bold underline">
        Faces : <br />
        {faces.map((face) => {
          return <></>;
        })}
        <br />
        <button
          className="px-4 py-2 font-semibold text-sm bg-sky-500 text-white rounded-none shadow-sm"
          onClick={() => {
            socket.emit("forceFresh");
          }}
        >
          Force refresh
        </button>
      </h1>
    </>
  );
}

export default TestRoom;
