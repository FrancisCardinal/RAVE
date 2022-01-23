import FormControl from '@mui/material/FormControl';
import Box from '@mui/material/Box';
import Select from '@mui/material/Select'
import InputLabel from '@mui/material/InputLabel'
import MenuItem from '@mui/material/MenuItem';
import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';

function LanguageSelection({ className }) {
	const [t, i18n] = useTranslation('common');
	const [language, setLanguage] = useState('');

	const handleChange = (event) => {
		setLanguage(event.target.value);
		i18n.changeLanguage(event.target.value);
	};
	
	return (
		<div className={className}>
			<Box sx={{ minWidth:120 }}>
			<FormControl fullWidth color='error'>
				<InputLabel color="error" id="language-select-label">{t('settingsPage.language.label')}</InputLabel>
				<Select
					labelId='language-select-label'
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