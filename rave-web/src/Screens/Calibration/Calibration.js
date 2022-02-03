import React from "react";
import { useTranslation } from "react-i18next";
import CalibStream from "../../Components/UI/CalibStream";
import CalibButton from "../../Components/UI/CalibButton";

function CalibrationScreen() {
  const [t] = useTranslation('common');
  return (
    <div>
      <h1 className="text-3xl mx-2 font-bold underline">{t('calibrationPage.title')}</h1>
      <CalibStream />
      <CalibButton />
    </div>
  );
}

export default CalibrationScreen;