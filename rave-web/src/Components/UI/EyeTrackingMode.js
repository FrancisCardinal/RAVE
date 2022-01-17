import Switch from '@mui/material/Switch'
import { useState } from 'react';
function EyeTrackingMode() {
    const [eye, setEyeTracking] = useState(false);

    const handleChange = (event) => {
        setEyeTracking(event.target.checked);
        console.log(eye);
    };
    return(
    <div className="flex bg-grey rounded-lg m-2 items-center w-min">
        <div><h1 className='p-2 font-medium w-max'>Mode Eye-tracking</h1></div>
        <div><Switch 
                className='align-middle' 
                color='error' 
                checked={eye} 
                onChange={handleChange}
            />
        </div>
    </div>);
}

export default EyeTrackingMode;