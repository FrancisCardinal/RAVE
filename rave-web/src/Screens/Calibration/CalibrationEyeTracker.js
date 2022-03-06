import React, { useContext, useEffect } from "react";
import { useTranslation } from "react-i18next";
import CalibConfigs from "../../Components/UI/CalibConfigs";
import SocketContext from "../../socketContext";

function EyeTrackerCalibScreen() {
  const ws = useContext(SocketContext);
  const [t] = useTranslation('common');

  // TO-DO: Find a solution to replace this useEffect so that it doesn't give a warning.
  useEffect(() => {
    ws && ws.emit("goToEyeTrackerCalib");
  }, []);
  
  return (
    <div>
      <h1 className="text-3xl mx-2 font-bold text-center underline">
        {t('eyeTrackerCalibrationPage.title')}
      </h1>
      <CalibConfigs />
    </div>
  );
}

export default EyeTrackerCalibScreen;