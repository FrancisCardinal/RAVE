import React, { useContext, useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import CalibStream from "../../Components/UI/CalibStream";
import CalibButton from "../../Components/UI/CalibButton";
import TextField from "@mui/material/TextField";
import SocketContext from "../../socketContext";
import { ErrorIcon } from "../../Ressources/icons";

function CalibrationScreen() {
  const ws = useContext(SocketContext);
  const [t] = useTranslation('common');
  const [errorMessage, setErrorMessage] = useState('');
  const [nbPoints, setNbPoints] = useState(5);
  const [orderPoly, setOrderPoly] = useState(3);
  const handlePointsChange = (event) => {
    setNbPoints(event.target.value);
  };
  const handleOrderChange = (event) => {
    setOrderPoly(event.target.value);
  };

  useEffect(() => {
    if (ws) {
      ws.on('newErrorMsg', (newErrorMessage) => {
        setErrorMessage(newErrorMessage);
      });
    }
  }, [ws]);

  return (
    <div className="flex flex-col">
      <h1 className="text-3xl mx-2 font-bold underline">{t('calibrationPage.title')}</h1>
      <div className="flex flex-row">
        <CalibStream />
        <div className="flex flex-col border-2 rounded-md p-2 h-min shadow-md border-grey">
          <CalibButton />
          <TextField error id="nb_points" 
                    label={t('calibrationPage.pointsLabel')}
                    defaultValue={5}
                    value={nbPoints}
                    onChange={handlePointsChange}
                    margin="normal"/>
          <TextField error id="order_polynom" 
                    label={t('calibrationPage.orderLabel')}
                    defaultValue={3}
                    value={orderPoly}
                    onChange={handleOrderChange}
                    margin="normal"/>
          <div className='border p-2 my-2 border-red rounded'>
            <ErrorIcon className={"w-6 h-6"}/>
            {errorMessage}
          </div>
          <button
              className="px-4 py-2 mt-2 font-semibold text-sm bg-grey text-black rounded-md shadow-sm"
              onClick={() => {
                ws.emit('changeCalibParams', {number:nbPoints, order:orderPoly});
                setErrorMessage('');
              }}
            >
              {t('calibrationPage.confirm')}
          </button>
          
        </div>
      </div>
    </div>
  );
}

export default CalibrationScreen;