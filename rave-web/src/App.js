import { useEffect, useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import HomeScreen from './Screens/Home';
import SettingsScreen from './Screens/Settings';
import HelpScreen from './Screens/Help';
import TestRoom from './Screens/Test';
import { Navbar } from './Components/UI';
import { SocketProvider } from './socketContext';
import io from 'socket.io-client';

function App() {
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    const ws = io('ws://localhost:9000');
    ws.on('connect', () => {
      console.log('Websocket connected');
      setSocket(ws);
    });
    ws.on('connect_error', (err) => {
      console.error('Failed to connect to websocket server : ', err.message);
    });
    return () => {
      ws.close();
    };
  }, []);

  return (
    <>
      <SocketProvider value={socket}>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<HomeScreen />} />
            <Route path="/settings" element={<SettingsScreen />} />
            <Route path="/help" element={<HelpScreen />} />
            <Route path="/serverTest" element={<TestRoom />} />
          </Routes>
          <Navbar />
        </BrowserRouter>
      </SocketProvider>
    </>
  );
}

export default App;
