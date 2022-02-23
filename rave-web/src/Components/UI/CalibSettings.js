import React, { useContext, useState, useEffect } from "react";
import CalibButton from "../../Components/UI/CalibButton";
import TextField from "@mui/material/TextField";
import { styled } from "@mui/material/styles";
import { ErrorIcon } from "../../Ressources/icons";
import { useTranslation } from "react-i18next";
import SocketContext from "../../socketContext";

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

function CalibSettings() {
  const ws = useContext(SocketContext);
  const [t] = useTranslation('common');
  
  const [nbPoints, setNbPoints] = useState(5);
  const handlePointsChange = (event) => {
    setNbPoints(event.target.value);
  };

  const [orderPoly, setOrderPoly] = useState(3);
  
  const handleOrderChange = (event) => {
    setOrderPoly(event.target.value);
  };
  
  const [errorMessage, setErrorMessage] = useState('');
  useEffect(() => {
    if (ws) {
      ws.on('newErrorMsg', (newErrorMessage) => {
        setErrorMessage(newErrorMessage);
      });
    }
    return (() => {
      ws && ws.emit('quitCalibration');
    });
  }, [ws]);
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
                  ws.emit('changeCalibParams', {number:nbPoints, order:orderPoly});
                  setErrorMessage('');
                }}
              >
                {t('visionCalibrationPage.confirm')}
            </button>
          </div>
  );
}

export default CalibSettings;