import { useTranslation } from "react-i18next";
import ConnectionStatus from "../../Components/UI/ConnectionStatus";
import LanguageSelection from "../../Components/UI/LanguageSelection";
import { Link } from 'react-router-dom';
import React from "react";

function SettingsScreen() {
  const [t] = useTranslation('common');

  return (
    <div>
      <h1 className="text-3xl ml-4 font-bold underline">{t('settingsPage.title')}</h1>
      <LanguageSelection className={"py-4 mx-4 max-w-md"}/>
      <ConnectionStatus />
      <Link to={`/calibration`} className="flex flex-row border border-grey max-w-md pl-3 mt-4 py-4 mx-4 rounded hover:border-black">{t('settingsPage.calibration')}</Link>
    </div>
  );
}

export default SettingsScreen;
