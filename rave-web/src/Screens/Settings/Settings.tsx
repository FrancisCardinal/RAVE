import { useTranslation } from "react-i18next";
import ConnectionStatus from "../../Components/UI/ConnectionStatus";
import LanguageSelection from "../../Components/UI/LanguageSelection";
import { Link } from 'react-router-dom';
import React, { useContext } from "react";
import Switch from '@mui/material/Switch';
import { DebugContext } from "../../DebugContextProvider";
import VisionModeSelection from "../../Components/UI/VisionModeSelection";

function SettingsScreen() {
  const [t] = useTranslation('common');
  const { debugging, toggleDebugging} = useContext(DebugContext);

  return (
    <div className="flex flex-col items-center">
      <h1 className="text-3xl font-bold underline text-center">{t('settingsPage.title')}</h1>
      <LanguageSelection className={"py-4 mx-4 w-3/4"}/>
      <VisionModeSelection />
      <ConnectionStatus/>
      <Link 
        to={`/calibration`}
        className="border border-grey p-2 w-3/4 mt-4 py-4 rounded hover:border-black">{t('settingsPage.visionCalibration')}
      </Link>
      <Link 
        to={`/calibration-eye-tracker`}
        className="border border-grey p-2 w-3/4 mt-4 py-4 rounded hover:border-black">{t('settingsPage.eyeTrackerCalibration')}
      </Link>
      <div className="border border-grey p-2 w-3/4 mt-4 py-4 rounded hover:border-black">
        {t('settingsPage.debugMode')}
        <Switch className="align-middle" color="error" checked={debugging} onChange={toggleDebugging} />
      </div>
    </div>
  );
}

export default SettingsScreen;
