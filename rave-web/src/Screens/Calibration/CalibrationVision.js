import React, { useContext, useEffect } from "react";
import { useTranslation } from "react-i18next";
import CalibStream from "../../Components/UI/CalibStream";
import SocketContext from "../../socketContext";
import { BrowserView, MobileView } from 'react-device-detect';
import CalibSettings from "../../Components/UI/CalibSettings";



function CalibrationScreen() {
  const ws = useContext(SocketContext);
  const [t] = useTranslation('common');

  useEffect(() => {
    ws && ws.emit("goToVisionCalib");
  }, []);

  return (
    <div className="flex flex-col">
      <h1 className="text-3xl mx-2 font-bold underline">{t('visionCalibrationPage.title')}</h1>
      <BrowserView>
        <div className="flex flex-row">
          <CalibStream />
          <CalibSettings />
        </div>
      </BrowserView>
      <MobileView>
      <div className="flex flex-col">
          <CalibStream />
          <CalibSettings />
        </div>
      </MobileView>
      
    </div>
  );
}

export default CalibrationScreen;