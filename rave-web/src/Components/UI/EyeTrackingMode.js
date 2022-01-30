import Switch from '@mui/material/Switch'
import React, { useContext, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import SocketContext from '../../socketContext';

function EyeTrackingMode() {
	const ws = useContext(SocketContext);
	const [t] = useTranslation('common');
	const [eye, setEyeTracking] = useState(false);

	const handleChange = (event) => {
		setEyeTracking(event.target.checked);
		ws.emit('activateEyeTracking', event.target.checked);
		console.log(event.target.checked);
	};

	useEffect(() => {
		if (ws) {
			ws.on();
		}
	}, [ws]);

	return(
	<div className="flex bg-grey rounded-lg m-2 items-center w-min">
		<h1 className='p-2 font-medium w-max'>{t('homePage.eyeTrackingLabel')}</h1>
		<div>
			<Switch 
				className='align-middle' 
				color='error' 
				checked={eye} 
				onChange={handleChange}
			/>
		</div>
	</div>);
}

export default EyeTrackingMode;