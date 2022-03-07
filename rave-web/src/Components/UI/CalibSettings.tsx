import React, { useState } from 'react';
import CalibButton from './CalibButton';
import TextField from '@mui/material/TextField';
import { styled } from '@mui/material/styles';
import { ErrorIcon } from '../../Ressources/icons';
import { useTranslation } from 'react-i18next';
import { useEmit, useEventListener } from '../../Hooks';
import { CLIENT_EVENTS, ChangeVisionCalibrationParamsEvent } from 'rave-protocol';

const CustomTextField = styled(TextField)({
  '& label.Mui-focused': {
    color: 'red',
  },
  '& .MuiOutlinedInput-root': {
    '&.Mui-focused fieldset': {
      borderColor: 'red',
    },
  },
});

/**
 * This component lets the user change the visual-audio calibration settings
 * (number of points and the polynomial). It also displays all the error messages concerning the calibration.
 */
function CalibSettings() {
  const emit = useEmit();
  const [t] = useTranslation('common');
  const [errorMessage, setErrorMessage] = useState('');
  const [nbPoints, setNbPoints] = useState(5);
  const [orderPoly, setOrderPoly] = useState(3);

  useEventListener(CLIENT_EVENTS.CALIBRATION_ERROR, ({ message }) => {
    setErrorMessage(message);
  });

  const handlePointsChange = (event : React.ChangeEvent<HTMLInputElement>) => {
    setNbPoints(Number(event.target.value));
  };

  const handleOrderChange = (event : React.ChangeEvent<HTMLInputElement>) => {
    setOrderPoly(Number(event.target.value));
  };

  return (
    <div className="flex flex-col border-2 rounded-md p-2 mx-4 mt-2 h-min shadow-md border-grey">
      <CalibButton />
      <CustomTextField id="nb_points" 
        label={t('visionCalibrationPage.pointsLabel')}
        defaultValue={5}
        value={nbPoints}
        onChange={handlePointsChange}
        margin="normal"/>
      <CustomTextField id="order_polynom" 
        label={t('visionCalibrationPage.orderLabel')}
        defaultValue={3}
        value={orderPoly}
        onChange={handleOrderChange}
        margin="normal"/>
      <div className='border p-2 my-2 border-red rounded'>
        <ErrorIcon className={"w-6 h-6"}/>
        <p className="text-red">{errorMessage}</p>
      </div>
      <button
        className="px-4 py-2 mt-2 font-semibold text-sm bg-grey text-black rounded-md shadow-sm"
        onClick={() => {
          emit(ChangeVisionCalibrationParamsEvent(nbPoints,orderPoly));
          setErrorMessage('');
        }}
      >
        {t('visionCalibrationPage.confirm')}
      </button>
    </div>
  );
}

export default CalibSettings;
