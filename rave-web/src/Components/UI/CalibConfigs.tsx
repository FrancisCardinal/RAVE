import React, { useState } from "react";
import AddCalibConfigs from "./AddCalibConfigs";
import List from "@mui/material/List";
import { DeleteIcon } from "../../Ressources/icons";
import IconButton from "@mui/material/IconButton";
import { ListItem } from "@mui/material";
import { useTranslation } from "react-i18next";
import { useEventListener, useEmit } from "../../Hooks";
import { CLIENT_EVENTS } from 'rave-protocol/clientEvents';
import { DeleteConfigEvent, EyeTrackingConfigSelectedEvent } from 'rave-protocol/pythonEvents';

function CalibConfigs() {
  const [t] = useTranslation("common");
  const emit = useEmit();

  const [configs, setConfigs] = useState<{name : string}[]>([]);
  
  const handleSelect = (name : string) => {
    var ptag = document.getElementById('selection-text');
    ptag && (ptag.innerHTML = name);
    emit(EyeTrackingConfigSelectedEvent(name));
  }

  const deleteConfig = (name : string) => {
    emit(DeleteConfigEvent(name));
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
              </div>
          </ListItem>)}
        </List>
      </div>
    </div>
  );
}

export default CalibConfigs;