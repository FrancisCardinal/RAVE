import Switch from '@mui/material/Switch'
import React, { useContext, useState } from 'react';
import { useTranslation } from 'react-i18next';
import SocketContext from '../../socketContext';

/**
 * This component is a switch to activate and deactivate the eye-tracking mode
 */
function EyeTrackingMode() {
	const ws = useContext(SocketContext);
	const [t] = useTranslation('common');
	const [eye, setEyeTracking] = useState(false);

	const handleChange = (event) => {
		setEyeTracking(event.target.checked);
		ws.emit('activateEyeTracking', event.target.checked);
	};

	return(
	<div className="flex bg-grey rounded-lg m-2 items-center w-min">
		<h1 className='p-2 font-medium w-max'>{t('homePage.eyeTrackingLabel')}</h1>
		<Switch 
			className='align-middle' 
			color='error' 
			checked={eye} 
			onChange={handleChange}
		/>
	</div>);
}

export default EyeTrackingMode;