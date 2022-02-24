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
        console.log("Status changing")
        setConnectionStatus(newStatus);
      });
    }
  }, [ws]);

  function StatusIcon() {
    if (connectionStatus === 0) {
      return(<NoWifiConnectionIcon className={"w-6 h-6"} />);
    }
    else if (connectionStatus === 2) {
      return(<WifiCheckedIcon className={"w-6 h-6"} />);
    }
  }

  return (
    <div className="flex flex-row border w-3/4 border-grey px-2 py-4 rounded hover:border-black">
        <h1 className="pr-2">{t('settingsPage.connectionLabel')}</h1>
        <StatusIcon /> 
    </div>
  );
}

export default ConnectionStatus;