import RAVE from '../../Ressources/RAVE.png'
import { Link } from 'react-router-dom';

function DesktopMainBar() {
	return (
	<nav className='flex flex-row bg-grey text-black font-mono inset-x-0 top-0 items-center'>
		<img className='w-10 h-full' src={RAVE} alt="raves logo"/>
		<h1 className=' left-0 pl-1 pr-3 text-2xl tracking-widest font-bold'>RAVE</h1>
		<Link 
			className='font-bold px-3 hover:font-medium'
			to={`/`}
		>
			Home
		</Link>
		<Link className='font-bold hover:font-medium px-3' to={`/settings`}>
			Settings
		</Link>
		<Link className='font-bold hover:font-medium px-3' to={`/help`}>
			Help
		</Link>
	</nav>
	);
}

export default DesktopMainBar;