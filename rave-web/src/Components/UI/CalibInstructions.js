
import { Modal, TextField } from "@mui/material";
import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { SaveIcon } from "../../Ressources/icons";
import PropTypes from "prop-types";
import { styled } from "@mui/material/styles";

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
}
function CalibInstructions({setInstructionModalOpen}) {
  const gifs = [
    "https://giphy.com/embed/GJi6ZBzgkWNmU",
    "https://giphy.com/embed/l41YdAa3Yll5NHfwI",
    "https://giphy.com/embed/65QZtTQC06Ot08sf50",
  ]
  const [t] = useTranslation('common');
  const [step, setStep] = useState(0);
  const nextStep = () => {
    const newStep = step + 1;
    setStep(newStep);
  }

  const [open, setOpen] = useState(false);
  const handleClose = () => setOpen(false);
  useEffect(() => {
    if (step >= 3) {
      setOpen(true);
    }
  }, [step]);

  const [name_id, setName_id] = useState("");
  const handleNameIdChange = event => {
    setName_id(event.target.value);
  };
  const handleSubmit = event => {
    setInstructionModalOpen(false);
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
        <div className="flex shadow-lg bg-red rounded w-max p-2 absolute left-1/2 top-1/2 -translate-y-1/2 -translate-x-1/2">
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
          </form>
        </div>
      </Modal>
    </div>

  );
}

export default CalibInstructions;