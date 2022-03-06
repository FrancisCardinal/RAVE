import React, { useState } from 'react';
import { MuteIcon, VolumeUp } from '../../Ressources/icons';
import IconButton from '@mui/material/Button';
import { useEmit } from '../../Hooks';
import { MuteRequestEvent } from 'rave-protocol/pythonEvents';

function MuteButton() {
  const emit = useEmit();
  const [soundOn, setSound] = useState(true);

  const handleClick = () => {
    setSound(!soundOn);
    emit(MuteRequestEvent(soundOn));
  };

  function SoundIcon() {
    if (soundOn) {
      return <VolumeUp className={'w-9 h-9'} />;
    } else {
      return <MuteIcon className={'w-9 h-9 mx-auto'} />;
    }
  }

  return (
    <div className="flex w-min rounded-md h-min m-2 center-items justify-center">
      <IconButton onClick={handleClick} aria-label="mute" variant="contained" color="error" size="small">
        <SoundIcon />
      </IconButton>
    </div>
  );
}

export default MuteButton;
