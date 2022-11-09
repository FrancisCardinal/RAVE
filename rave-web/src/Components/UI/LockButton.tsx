import React, { useState } from 'react';
import { LockIcon, UnlockedIcon } from '../../Ressources/icons';
import IconButton from '@mui/material/Button';
import { useEmit } from '../../Hooks';

/**
 * This component allows the user to lock ALL target changes
 */
function LockButton() {
  const emit = useEmit();
  const [locked, setLocked] = useState(false);

  const handleClick = () => {
    setLocked(!locked);
  };

  function LockButtonIcon() {
    if (locked) {
      return <LockIcon className={'w-9 h-9'} />;
    } else {
      return <UnlockedIcon className={'w-9 h-9'} />;
    }
  }

  return (
    <div className="flex w-min rounded-md h-min m-2 center-items justify-center">
      <IconButton onClick={handleClick} aria-label="mute" variant="contained" color={ locked ? "error" : "success"} size="small">
        <LockButtonIcon />
      </IconButton>
    </div>
  );
}

export default LockButton;
