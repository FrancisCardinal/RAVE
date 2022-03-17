import React, { useState } from "react";
import Fab from "@mui/material/Fab";
import { AddIcon } from "../../Ressources/icons";
import CalibInstructions from "./CalibInstructions";
import { Modal } from "@mui/material";
import { BrowserView, MobileView } from 'react-device-detect';
import { useEmit } from "../../Hooks";
import { StartEyeTrackerCalibrationEvent } from 'rave-protocol/pythonEvents';

/**
 * This component is a button to create a new eye-tracking calibration configuration.
 * When it's clicked, a modal with the instructions to follow appear.
 */
const AddCalibConfigs = () => {
  const emit = useEmit();
  const [open, setOpen] = useState(false);
  const handleClose = () => setOpen(false);
  const handleOpen = () => {
    setOpen(true);
    emit(StartEyeTrackerCalibrationEvent());
  }

  return (
    <div className="absolute border-l  border-grey right-0 inset-y-0 p-2">
      {/* @ts-ignore */}
      <Fab size="small" color="error" aria-label="add" onClick={handleOpen}>
        <AddIcon className={"w-5 h-5"} />
      </Fab>
      <MobileView>
        <Modal
          open={open}
          onClose={handleClose}
          aria-labelledby="modal-create-config"
          aria-describedby="modal-create-description"
        >
          <div className="absolute bg-white shadow-md shadow-red rounded-md w-fit p-4 left-1/2 top-1/2 -translate-y-1/2 -translate-x-1/2">
            <CalibInstructions setInstructionModalOpen={setOpen} />
          </div>
        </Modal>
      </MobileView>
      <BrowserView>
        <Modal
          open={open}
          onClose={handleClose}
          aria-labelledby="modal-create-config"
          aria-describedby="modal-create-description"
        >
          <div className="absolute bg-white shadow-md shadow-red rounded-md w-3/4 h-3/4 p-4 left-1/2 top-1/2 -translate-y-1/2 -translate-x-1/2">
            <CalibInstructions setInstructionModalOpen={setOpen} />
          </div>
        </Modal>
      </BrowserView>
    </div>
  );
}

export default AddCalibConfigs;