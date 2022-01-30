import CircularProgress from "@mui/material/CircularProgress";
import React, { useContext, useEffect, useState } from "react";
import { useTranslation } from "react-i18next"
import { WifiCheckedIcon, NoWifiConnectionIcon } from "../../Ressources/icons";
import SocketContext from "../../socketContext";

function ConnectionStatus() {
  const ws = useContext(SocketContext);
  const [t] = useTranslation('common'); 
  const [connectionStatus, setConnectionStatus] = useState(0);
  
  useEffect(() => {
    if (ws) {
      ws.on('getConnectionStatus', (newStatus) => {
        setConnectionStatus(newStatus);
      });
      return () => {
        ws.close();
      };
    }
  }, [ws]);

  function StatusIcon() {
    if (connectionStatus === 0) {
      return(<NoWifiConnectionIcon className={"w-6 h-6"} />);
    }
    else if (connectionStatus === 1) {
      return(<CircularProgress color="error" size={25} thickness={5} />);
    }
    else if (connectionStatus === 2) {
      return(<WifiCheckedIcon className={"w-6 h-6"} />);
    }
  }

  return (
    <div className="flex flex-row border border-grey max-w-md pl-3 py-4 mx-4 rounded hover:border-black">
        <h1 className="pr-4">{t('settingsPage.connectionLabel')}</h1>
        <StatusIcon /> 
    </div>
  );
}

export default ConnectionStatus;