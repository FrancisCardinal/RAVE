import FormControl from '@mui/material/FormControl';
import Box from '@mui/material/Box';
import Select from '@mui/material/Select'
import InputLabel from '@mui/material/InputLabel'
import MenuItem from '@mui/material/MenuItem';
import { useState, useEffect } from 'react';

function LanguageSelection() {
	const [language, setLanguage] = useState('');

	const handleChange = (event) => {
		setLanguage(event.target.value);
	};
	useEffect(() => {
		console.log(language);
		return () => {
			console.log("Destructeur:", language);
		}
	}, [language])
	return (
		<Box sx={{ minWidth:120 }}>
			<FormControl fullWidth color='error'>
				<InputLabel color="error" id="language-select-label">Language</InputLabel>
				<Select
					labelId='language-select-label'
					id="language-select"
					value={language}
					label="Language"
					onChange={handleChange}
				>
					<MenuItem value={0}>French</MenuItem>
					<MenuItem value={1}>English</MenuItem>
				</Select>
			</FormControl>
		</Box>
	);
}

export default LanguageSelection;