
import { Collapse, Modal, TextField } from "@mui/material";
import React, { useContext, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { SaveIcon } from "../../Ressources/icons";
import PropTypes from "prop-types";
import { styled } from "@mui/material/styles";
import SocketContext from "../../socketContext";

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

CalibInstructions.propTypes = {
  setInstructionModalOpen: PropTypes.func,
  name_history: PropTypes.array,
}

/**
 * This component describes the steps to follow to calibrate the eye-tracker camera for the user
 * and pops a form to save the calibratio once it's done.
 * The steps are described by gifs showing the eye movement need.
 */
function CalibInstructions({setInstructionModalOpen, name_history}) {
  const gifs = [
    "https://giphy.com/embed/GJi6ZBzgkWNmU",
    "https://giphy.com/embed/l41YdAa3Yll5NHfwI",
    "https://giphy.com/embed/65QZtTQC06Ot08sf50",
  ]
  const ws = useContext(SocketContext);
  const [t] = useTranslation('common');
  const [step, setStep] = useState(0);
  const nextStep = () => {
    const newStep = step + 1;
    setStep(newStep);
    ws.emit("nextCalibStep");
  }

  const [error_open, setError_open] = useState(false);
  const [open, setOpen] = useState(false);
  const handleClose = () => setOpen(false);
  useEffect(() => {
    if (step >= 3) {
      setOpen(true);
      document.getElementById('next-button').disabled = true;
    }
  }, [step]);

  const [name_id, setName_id] = useState("");
  const handleNameIdChange = event => {
    setName_id(event.target.value);
    setError_open(false);
  };

  /**
   * This function is called when the user clicks on the save button of the saving form.
   * It verifies if the input value is already a calibration files and displays an error if it is.
   * If the name is valid, the saving form and instructions are close and it sends the new 
   * calibartion to the server.
   */
  const handleSubmit = event => {
    event.preventDefault();
    if (!name_history.find((element) => element.name === name_id))
    {
      setOpen(false);
      setInstructionModalOpen(false);
      ws.emit("addNewConfig", name_id);
    }
    else 
    {
      setError_open(true);
    }
  };

  return (
    <div className="h-full">
      <p className="bg-grey w-fit rounded p-2 shadow">{t('eyeTrackerCalibrationPage.instruction')}</p>
      <div className="flex justify-center h-full pb-8">
        <iframe className="p-2 justify-center" width="100%" height="100%" title="moving-eye" src={gifs[step]}></iframe>
      </div>
      <button
        id="next-button"
        className="absolute bottom-0 right-0 px-4 m-4 py-2 font-semibold text-sm bg-grey text-black rounded-md shadow-sm"
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
        <div className="flex shadow-lg bg-white rounded w-fit p-2 absolute left-1/2 top-1/2 -translate-y-1/2 -translate-x-1/2">
          <form onSubmit={handleSubmit} id="new-calib-config" className="flex flex-col content-center">
            <h1 className="font-bold underline text-center decoration-2 w-full">{t('eyeTrackerCalibrationPage.modalTitle')}</h1>
            <div className="flex flex-row pt-2 justify-center">
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