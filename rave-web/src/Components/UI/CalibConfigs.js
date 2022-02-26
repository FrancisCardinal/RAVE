import React, { useContext, useEffect, useState } from "react";
import AddCalibConfigs from "./AddCalibConfigs";
import List from "@mui/material/List";
import { DeleteIcon } from "../../Ressources/icons";
import IconButton from "@mui/material/IconButton";
import { ListItem } from "@mui/material";
import SocketContext from "../../socketContext";
import { useTranslation } from "react-i18next";

function CalibConfigs() {
  const [t] = useTranslation("common");
  const ws = useContext(SocketContext);
  const [configs, setConfigs] = useState([]);
  
  const handleSelect = (name) => {
    var ptag = document.getElementById('selection-text');
    ptag.innerHTML = name;
    ws.emit("eyeTrackingConfigSelected", name);
  }

  const deleteConfig = (id) => {
    ws.emit("deleteConfig", (id));
  }
  
  useEffect(() => {
    if (ws) {
      ws.on('onConfigList', (configList) => {
        setConfigs(configList);
      });
    }
  }, [ws]);

  return (
    <div>
      <div className="flex relative border m-2 p-4 border-grey rounded">
        <div className="w-fit">
          <p id="selection-text">{t('eyeTrackerCalibrationPage.placeholder')}</p>
        </div>
        <AddCalibConfigs name_history={configs}/>
      </div>
      <div className="bg-grey rounded m-2">
        <List>
          {configs.map((item) => <ListItem
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