import NavMenu from './NavMenu'
import RAVE from '../../Ressources/RAVE.png'

export function MobileMainBar() {
  return (
    <div className="flex flex-row bg-grey text-black font-mono h-14 inset-x-0 top-0 items-center">
      <img className='w-14 h-full' src={RAVE} alt="raves logo"/>
      <NavMenu className={"absolute px-2 right-0"}/>
    </div>
  );
}

export default MobileMainBar;
