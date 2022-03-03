import React, { useState } from "react";
import Fab from "@mui/material/Fab";
import { AddIcon } from "../../Ressources/icons";
import CalibInstructions from "./CalibInstructions";
import { Modal } from "@mui/material";
import { BrowserView, MobileView } from 'react-device-detect';
import SocketContext from "../../socketContext";
import PropTypes from "prop-types";

AddCalibConfigs.propTypes = {
  name_history: PropTypes.array,
}
function AddCalibConfigs({name_history}) {
  const ws = useContext(SocketContext);
  const [open, setOpen] = useState(false);
  const handleClose = () => setOpen(false);
  const handleOpen = () => {
    setOpen(true);
    ws.emit("startEyeTrackerCalib");
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
          <div className="absolute bg-grey shadow-lg shadow-red rounded-md p-4 w-11/12 h-3/4 left-1/2 top-1/2 -translate-y-1/2 -translate-x-1/2">
            <CalibInstructions name_history={name_history} setInstructionModalOpen={setOpen} />
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
          <div className="absolute bg-grey shadow-lg rounded-md p-4 w-3/4 h-3/4 left-1/2 top-1/2 -translate-y-1/2 -translate-x-1/2">
            <CalibInstructions name_history={name_history} setInstructionModalOpen={setOpen} />
          </div>
        </Modal>
      </BrowserView>
    </div>
  );
}

export default AddCalibConfigs;