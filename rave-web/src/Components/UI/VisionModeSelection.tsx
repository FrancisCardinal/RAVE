import React, { FC, useState } from 'react';
import Box from '@mui/material/Box';
import { FormControl, InputLabel, MenuItem, Select, SelectChangeEvent } from '@mui/material';
import { useTranslation } from 'react-i18next';
import { useEmit } from '../../Hooks/useEmit';
import { ChangeVisionModeEvent } from 'rave-protocol';

const VisionModeSelection : FC = () => {
  const [t] = useTranslation('common');
  const emit = useEmit();

  const [visionMode, setVisionMode] = useState('mute');
  const handleChange= (event: SelectChangeEvent<string>) => {
    emit(ChangeVisionModeEvent(event.target.value));
    setVisionMode(event.target.value);
  }
  return (
    <div className='pb-4 mx-4 w-3/4'>
      <Box sx={{ minWidth: 120 }}>
        <FormControl fullWidth color="error">
          <InputLabel color='error' id="vision-mode-selection-label">
            {t("settingsPage.visionMode.label")}
          </InputLabel>
          <Select
            labelId='vision-mode-select-label'
            id='vision-mode-select'
            value={visionMode}
            label='Vision Mode'
            onChange={handleChange}
          >
            <MenuItem value={'mute'}>{t('settingsPage.visionMode.muteMode')}</MenuItem>
            <MenuItem value={'hear'}>{t('settingsPage.visionMode.hearMode')}</MenuItem>
          </Select>
        </FormControl>
      </Box>
    </div>
  );
}

export default VisionModeSelection;