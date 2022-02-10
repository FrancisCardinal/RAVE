import RAVE from '../../Ressources/RAVE.png'
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import React, { useContext } from 'react';
import SocketContext from '../../socketContext';

function DesktopMainBar() {
	const ws = useContext(SocketContext);
	const [t] = useTranslation('common');
	return (
	<nav className='flex flex-row bg-grey text-black font-mono inset-x-0 top-0 items-center'>
		<img className='w-10 h-full' src={RAVE} alt="raves logo"/>
		<h1 className=' left-0 pl-1 pr-3 text-2xl tracking-widest font-bold'>RAVE</h1>
		<Link 
			className='font-bold px-3 hover:font-medium'
			to={`/`}
			onClick={() => {
        ws.emit("quitCalibration");
      }} 
		>
			{t('navigationBar.homePage')}
		</Link>
		<Link className='font-bold hover:font-medium px-3' 
			onClick={() => {
        ws.emit("quitCalibration");
      }} 
			to={`/settings`}>
				{t('navigationBar.settingsPage')}
		</Link>
		<Link 
			className='font-bold hover:font-medium px-3' 
			to={`/help`}
			onClick={() => {
        ws.emit("quitCalibration");
      }} 
		>
		{t('navigationBar.helpPage')}
		</Link>
	</nav>
	);
}

export default DesktopMainBar;