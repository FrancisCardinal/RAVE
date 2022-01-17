import RAVE from '../../Ressources/RAVE.png'

function MainBar() {
    return (<div className='flex flex-row bg-grey text-black font-mono inset-x-0 top-0 items-center'>
        <img className='w-10 h-10' src={RAVE} alt="raves logo"/>
        <h1 className=' left-0 pl-24 text-2xl font-bold'>RAVE</h1>
        <p className='font-mono'>-Rehaussement de parole Vidéo et Audio par video avec écouteurs intelligent</p>
    </div>);
}

export default MainBar;