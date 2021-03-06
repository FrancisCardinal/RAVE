import React, { useEffect, useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import HomeScreen from './Screens/Home';
import SettingsScreen from './Screens/Settings';
import HelpScreen from './Screens/Help';
import CalibrationScreen from './Screens/Calibration';
import { SocketProvider, WebSocketType } from './socketContext';
import { io } from "socket.io-client"
import DesktopMainBar from './Components/UI/DesktopMainBar';
import MobileMainBar from './Components/UI/MobileMainBar';
import { BrowserView, MobileView } from 'react-device-detect';
import EyeTrackerCalibScreen from './Screens/Calibration/CalibrationEyeTracker';
import DebugContextProvider from './DebugContextProvider';

function App() {
  const [socket, setSocket] = useState<WebSocketType | null>(null);
  useEffect(() => {
    if (process.env.REACT_APP_ONLINE_MODE === 'true') {
      let URL = window.location.origin;
      URL = URL.replace('http', 'ws');
      URL = URL.replace(':3000', ':9000');
      const ws = io(URL);
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
    <DebugContextProvider>
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
              <Route path="/calibration" element={<CalibrationScreen />} />
              <Route path="/calibration-eye-tracker" element={<EyeTrackerCalibScreen />} />
            </Routes>
          </BrowserRouter>
      </SocketProvider>
    </DebugContextProvider>
    </>
  );
}

export default App;
