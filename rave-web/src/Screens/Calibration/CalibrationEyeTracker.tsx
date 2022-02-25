import React, { useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import CalibConfigs from '../../Components/UI/CalibConfigs';
import { useEmit } from '../../Hooks';
import { GoToEyeTrackerCalibrationEvent } from 'rave-protocol/pythonEvents';

function EyeTrackerCalibScreen() {
  const emit = useEmit();
  const [t] = useTranslation('common');

  useEffect(() => {
    emit(GoToEyeTrackerCalibrationEvent());
  }, [emit]);

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
