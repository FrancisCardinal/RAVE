import React, { useState } from "react";
import AddCalibConfigs from "./AddCalibConfigs";
import List from "@mui/material/List";
import { DeleteIcon, NoIcon, YesIcon } from "../../Ressources/icons";
import { IconButton, Button } from "@mui/material";
import { ListItem, Modal } from "@mui/material";
import { useTranslation } from "react-i18next";
import { useEventListener, useEmit } from "../../Hooks";
import { CLIENT_EVENTS } from 'rave-protocol/clientEvents';
import { DeleteConfigEvent, EyeTrackingConfigSelectedEvent } from 'rave-protocol/pythonEvents';

/**
 * This component lists the eye-tracking calibration configurations available to select.
 * It has a button to add and create a new calibration and each configuration has a delete button.
 */
function CalibConfigs() {
  const [t] = useTranslation("common");
  const emit = useEmit();
  const [nameDelete, setNameDelete] = useState<string>("");
  const [configs, setConfigs] = useState<{name : string}[]>([]);
  
  const handleSelect = (name : string) => {
    var ptag = document.getElementById('selection-text');
    ptag && (ptag.innerHTML = name);
    emit(EyeTrackingConfigSelectedEvent(name));
  }
  const [open, setOpen] = useState(false);
  const handleClose = () => setOpen(false);
  const deleteConfig = (name:string) => {
    setOpen(true);
    setNameDelete(name);    
  }
  const deleteClick = (name : string) => {
    emit(DeleteConfigEvent(name));
    setOpen(false);
  }
  useEventListener(CLIENT_EVENTS.EYE_TRACKING_CONFIGURATIONS, ({configuration}) => {
    console.log("New configs");
    setConfigs(configuration);
  });

  return (
    <div>
      <div className="flex relative border m-2 p-4 border-grey rounded">
        <div className="w-fit">
          <p id="selection-text">{t('eyeTrackerCalibrationPage.placeholder')}</p>
        </div>
        <AddCalibConfigs />
      </div>
      <div className="bg-grey rounded m-2">
        <List>
          {configs?.map((item) => <ListItem
            key={item.name}
             sx={{hover: {fontWeight: "bold"}}}>
              <p className="hover:font-medium" onClick={() => handleSelect(item.name)}>{item.name}</p>  
              <div className=" absolute right-0 p-5">
              <IconButton edge="end" aria-label="delete" onClick={() => deleteConfig(item.name)}>
                <DeleteIcon className={"w-5 h-5"}/>
              </IconButton>
              <Modal
                open={open}
                onClose={handleClose}
                aria-labelledby="modal-delete-config"
                aria-describedby="modal-delete-description"
              >
                <div className="flex flex-col w-fit place-items-center rounded shadow-lg p-2 bg-white absolute left-1/2 top-1/2 -translate-y-1/2 -translate-x-1/2">
                  <p className="font-black">{t('eyeTrackerCalibrationPage.deleteMessage')}</p>
                  {nameDelete}
                  <div className="flex flex-row">
                    <Button sx={{ margin: '2px' }} onClick={() => setOpen(false)} variant="contained" color="error" size="small">
                      <NoIcon className={"w-5 h-5"} />
                    </Button>
                    <Button sx={{ margin: '2px' }} onClick={() => deleteClick(nameDelete)} variant="contained" color="success" size="small">
                      <YesIcon className={"w-5 h-5"} />
                    </Button>
                  </div>
                </div>
              </Modal>
              </div>
          </ListItem>)}
        </List>
      </div>
    </div>
  );
}

export default CalibConfigs;