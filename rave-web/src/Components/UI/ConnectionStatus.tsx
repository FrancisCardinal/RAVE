import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { WifiCheckedIcon, NoWifiConnectionIcon } from '../../Ressources/icons';
import { useEventListener, useEmit } from '../../Hooks';
import { CLIENT_EVENTS, GetPythonConnectionStatus, ConnectionStatusPayload } from 'rave-protocol';

/**
 * This component displays the current connection with the prototype.
 */

function ConnectionStatus() {
  const emit = useEmit();
  const [t] = useTranslation('common');
  const [connectionStatus, setConnectionStatus] = useState(0);

  useEventListener(CLIENT_EVENTS.CONNECTION_STATUS, (payload: ConnectionStatusPayload) => {
    console.log('Status changing');
    setConnectionStatus(payload.status);
  });

  useEffect(() => {
    emit(GetPythonConnectionStatus());
  }, [emit])

  // TODO : Query connection status on create with useEffect

  function StatusIcon() {
    if (connectionStatus === 0) {
      return <NoWifiConnectionIcon className={'w-6 h-6'} />;
    }
    return <WifiCheckedIcon className={'w-6 h-6'} />;
  }

  return (
    <div className="flex flex-row border w-3/4 border-grey px-2 py-4 rounded hover:border-black">
      <h1 className="pr-2">{t('settingsPage.connectionLabel')}</h1>
      <StatusIcon />
    </div>
  );
}

export default ConnectionStatus;
