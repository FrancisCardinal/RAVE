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
  const dummy_list = [
    {id: 1, name: 'Amélie Rioux-Joyal'},
    {id: 2, name: 'Jacob Kealy'},
    {id: 3, name: 'Jérémy Bélec'},
    {id: 4, name: 'Francis Cardinal'}
  ];
  const ws = useContext(SocketContext);
  const [configs, setConfigs] = useState(dummy_list);
  // const [selection, setSelection] = useState(  )
  
  const handleSelect = (name) => {
    // setSelection(name);
    var ptag = document.getElementById('selection-text');
    ptag.innerHTML = name;
    ws.emit("eyeTrackingConfigSelected", name);
  }

  const deleteConfig = (id) => {
    const new_lists = configs.filter(x => {
      return x.id !== id;
    })
    setConfigs(new_lists);
    ws.emit("deleteConfig");
  }
  
  useEffect(() => {
    if (ws) {
      ws.on('getEyeTrackingConfigs', (newconfigs) => {
        console.log("New configs");
        setConfigs(newconfigs);
      });
    }
  }, [ws]);

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
          {configs.map((item) => <ListItem
            key={item.id}
             sx={{hover: {fontWeight: "bold"}}}>
              <p className="hover:font-medium" onClick={() => handleSelect(item.name)}>{item.name}</p>  
              <div className=" absolute right-0 p-5">
                <IconButton edge="end" aria-label="delete" onClick={() => deleteConfig(item.id)}>
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