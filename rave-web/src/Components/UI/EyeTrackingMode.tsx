import Switch from '@mui/material/Switch';
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useEmit } from '../../Hooks';
import { ActivateEyeTrackingEvent } from 'rave-protocol/pythonEvents';

function EyeTrackingMode() {
  const [t] = useTranslation('common');
  const [eye, setEyeTracking] = useState(false);
  const emit = useEmit();

  const handleChange = (event : React.ChangeEvent<HTMLInputElement>) => {
    setEyeTracking(event.target.checked);
    emit(ActivateEyeTrackingEvent(event.target.checked));
  };

  return (
    <div className="flex bg-grey rounded-lg m-2 items-center w-min">
      <h1 className="p-2 font-medium w-max">{t('homePage.eyeTrackingLabel')}</h1>
      <div>
        <Switch className="align-middle" color="error" checked={eye} onChange={handleChange} />
      </div>
    </div>
  );
}

export default EyeTrackingMode;
