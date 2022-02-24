import React, { useContext, useEffect } from "react";
import { useTranslation } from "react-i18next";
import CalibConfigs from "../../Components/UI/CalibConfigs";
import SocketContext from "../../socketContext";

function EyeTrackerCalibScreen() {
  const ws = useContext(SocketContext);
  useEffect(() => {
    ws && ws.emit("goToEyeTrackerCalib");
  }, []);
  const [t] = useTranslation('common');
  return (
    <div>
      <h1 className="text-3xl mx-2 font-bold text-center underline">{t('eyeTrackerCalibrationPage.title')}</h1>
      <div>
        <CalibConfigs />
      </div>
    </div>
  );
}

export default EyeTrackerCalibScreen;