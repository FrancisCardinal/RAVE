import { useTranslation } from "react-i18next";
import ConnectionStatus from "../../Components/UI/ConnectionStatus";
import LanguageSelection from "../../Components/UI/LanguageSelection";
import React from "react";

function SettingsScreen() {
  const [t] = useTranslation('common');

  return (
    <div>
      <h1 className="text-3xl ml-4 font-bold underline">{t('settingsPage.title')}</h1>
      <LanguageSelection className={"py-4 mx-4 max-w-md"}/>
      <ConnectionStatus />
    </div>
  );
}

export default SettingsScreen;
