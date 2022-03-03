
import { Collapse, Modal, TextField } from "@mui/material";
import React, { FC, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { SaveIcon } from "../../Ressources/icons";
import { styled } from "@mui/material/styles";
import { useEmit } from "../../Hooks";
import { EyeTrackerNextCalibrationStepEvent, EyeTrackerAddNewConfigEvent } from 'rave-protocol/pythonEvents';


const CustomTextField = styled(TextField)({
  '& label.Mui-focused': {
    color: 'black',
  },
  '& .MuiOutlinedInput-root': {
    '&.Mui-focused fieldset': {
      borderColor: 'black',
    },
  },
});

const gifs = [
  "https://giphy.com/embed/GJi6ZBzgkWNmU",
  "https://giphy.com/embed/l41YdAa3Yll5NHfwI",
  "https://giphy.com/embed/65QZtTQC06Ot08sf50",
]
interface CalibInstructionsProps {
  setInstructionModalOpen: (openState : boolean) => void;
  name_history: {id : number, name : string}[],
}

const CalibInstructions : FC<CalibInstructionsProps> = ({setInstructionModalOpen, name_history}) => {
  const emit = useEmit();
  const [t] = useTranslation('common');
  const [step, setStep] = useState(0);
  const [open, setOpen] = useState(false);
  const handleClose = () => setOpen(false);
  const [name_id, setName_id] = useState("");
  const [error_open, setError_open] = useState(false);
  
  useEffect(() => {
    if (step >= 3) {
      setOpen(true);
    }
  }, [step]);

  const nextStep = () => {
    const newStep = step + 1;
    setStep(newStep);
    emit(EyeTrackerNextCalibrationStepEvent());
  }

  const handleNameIdChange = (event : React.ChangeEvent<HTMLInputElement>) => {
    setName_id(event.target.value);
    setError_open(false);
  };
  
  const handleSubmit = (_event : React.FormEvent<HTMLFormElement>) => {
    _event.preventDefault();
    if (!name_history.find((element) => element.name === name_id))
    {
      setOpen(false);
      setInstructionModalOpen(false);
      emit(EyeTrackerAddNewConfigEvent(name_id));
    }
    else 
    {
      setError_open(true);
    }
  };

  return (
    <div>
      <p className="bg-white w-fit rounded p-2 shadow">{t('eyeTrackerCalibrationPage.instruction')}</p>
      <div className="flex justify-center">
        <iframe className="p-2 justify-center" title="moving-eye" src={gifs[step]} width="480" height="298"></iframe>
      </div>
      <button
        className="absolute animate-pulse bottom-0 right-0 px-4 m-4 py-2 font-semibold text-sm bg-red text-black rounded-md shadow-sm"
        onClick={nextStep}
      >
        {t('eyeTrackerCalibrationPage.next')}
      </button>
      <Modal
        open={open}
        onClose={handleClose}
        aria-labelledby="modal-save-config"
        aria-describedby="modal-save-description"
      >
        <div className="flex shadow-lg bg-white rounded w-max p-2 absolute left-1/2 top-1/2 -translate-y-1/2 -translate-x-1/2">
          <form onSubmit={handleSubmit} id="new-calib-config">
            <h1 className="font-bold underline decoration-2 w-max">{t('eyeTrackerCalibrationPage.modalTitle')}</h1>
            <div className="flex flex-row pt-2">
              <CustomTextField
                label={t('eyeTrackerCalibrationPage.configName')}
                id="name"
                placeholder={t('eyeTrackerCalibrationPage.configPlaceholder')}
                variant="outlined"
                value={name_id}
                onChange={handleNameIdChange}
              />
              <button type="submit"><SaveIcon className={"w-6 h-6 ml-2"}/></button>
            </div>
            <Collapse in={error_open}>
              <p className="text-red text-xs relative w-fit">{t('eyeTrackerCalibrationPage.errorMessage')}</p>
            </Collapse>
          </form>
        </div>
      </Modal>
    </div>

  );
}

export default CalibInstructions;