import Stack from '@mui/material/Stack';
import { VolumeDown, VolumeUp } from '../../Ressources/icons';
import Slider from '@mui/material/Slider';
import React, { useContext, useState } from 'react';
import SocketContext from '../../socketContext';

function VolumeSlider() {
	const ws = useContext(SocketContext);
	const [value, setValue] = useState(30);

	const handleChange = (event, newValue) => {
		setValue(newValue);
		console.log(value);
		ws.emit('setVolume', event.target.value);
	};

	return(<div className='w-80'>
		<Stack spacing={2} direction="row" sx={{ mb: 1 }} alignItems="center">
			<VolumeDown className={"w-9 h-9"}/>
			<Slider aria-label="Volume" value={value} onChange={handleChange} size='medium' color='error'/>
			<VolumeUp className={"w-10 h-10"} />
		</Stack>
	</div>);
}

export default VolumeSlider;