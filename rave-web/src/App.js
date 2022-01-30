import React, { useEffect, useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import HomeScreen from './Screens/Home';
import SettingsScreen from './Screens/Settings';
import HelpScreen from './Screens/Help';
import TestRoom from './Screens/Test';
import { SocketProvider } from './socketContext';
import io from 'socket.io-client';
import DesktopMainBar from "./Components/UI/DesktopMainBar";
import MobileMainBar from './Components/UI/MobileMainBar';
import {BrowserView, MobileView} from 'react-device-detect';

function App() {
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    if (process.env.REACT_APP_ONLINE_MODE === 'true') {
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
    }
  }, []);

  return (
    <>
      <SocketProvider value={socket}>
        <BrowserRouter>
          <BrowserView>
            <DesktopMainBar />
          </BrowserView>
          <MobileView>
            <MobileMainBar />
          </MobileView>
          <Routes>
            <Route path="/" element={<HomeScreen />} />
            <Route path="/settings" element={<SettingsScreen />} />
            <Route path="/help" element={<HelpScreen />} />
            <Route path="/serverTest" element={<TestRoom />} />
          </Routes>
        </BrowserRouter>
      </SocketProvider>
    </>
  );
}

export default App;
