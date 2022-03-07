import FormControl from '@mui/material/FormControl';
import Box from '@mui/material/Box';
import Select, { SelectChangeEvent } from '@mui/material/Select';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import React, { useState, FC } from 'react';
import { useTranslation } from 'react-i18next';

interface LanguageSelectionProps {
  className: string,
}
/**
 * This component allows the user to change the site's language.
 * The current languages available are english and french.
 */
const LanguageSelection : FC<LanguageSelectionProps> = (props) => {
  const [t, i18n] = useTranslation('common');
  const [language, setLanguage] = useState(i18n.language);
  
  const handleChange = (event : SelectChangeEvent<string>) => {
    setLanguage(event.target.value);
    i18n.changeLanguage(event.target.value);
  };

  return (
    <div className={props.className}>
      <Box sx={{ minWidth: 120 }}>
        <FormControl fullWidth color="error">
          <InputLabel color="error" id="language-select-label">
            {t('settingsPage.language.label')}
          </InputLabel>
          <Select
            labelId="language-select-label"
            id="language-select"
            value={language}
            label="Language"
            onChange={handleChange}
          >
            <MenuItem value={'fr'}>{t('settingsPage.language.french')}</MenuItem>
            <MenuItem value={'en'}>{t('settingsPage.language.english')}</MenuItem>
          </Select>
        </FormControl>
      </Box>
    </div>
  );
}

export default LanguageSelection;
