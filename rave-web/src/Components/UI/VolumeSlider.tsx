import React, { useState } from 'react';
import Stack from '@mui/material/Stack';
import { VolumeDown, VolumeUp } from '../../Ressources/icons';
import Slider from '@mui/material/Slider';
import { useEmit } from '../../Hooks';
import { SetVolumeEvent } from 'rave-protocol/pythonEvents';

/**
 * This component allows the user to modify the headphone's volume via a slider.
 */
function VolumeSlider() {
  const emit = useEmit();
  const [value, setValue] = useState(30);

  const handleChange = (event : React.ChangeEvent<HTMLInputElement>, newValue : number) => {
    setValue(newValue);
    emit(SetVolumeEvent(Number(event.target.value)));
  };

  return (
    <div className="w-80">
      <Stack spacing={2} direction="row" sx={{ mb: 1 }} alignItems="center">
        <VolumeDown className={'w-9 h-9'} />
        {/* @ts-ignore */}
        <Slider aria-label="Volume" value={value} onChange={handleChange} size="medium" color="error" />
        <VolumeUp className={'w-10 h-10'} />
      </Stack>
    </div>
  );
}

export default VolumeSlider;
