import Switch from '@mui/material/Switch'
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

function EyeTrackingMode() {
	const [t, i18n] = useTranslation('common');
	const [eye, setEyeTracking] = useState(false);

	const handleChange = (event) => {
		setEyeTracking(event.target.checked);
		console.log(event.target.checked);
	};
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