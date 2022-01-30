import { MuteIcon, VolumeUp } from "../../Ressources/icons";
import IconButton from "@mui/material/Button";
import React, { useContext, useEffect, useState } from "react";
import SocketContext from "../../socketContext";

function MuteButton() {
	const ws = useContext(SocketContext);
	const [soundOn, setSound] = useState(true);
	const handleClick = () => {
		setSound(!soundOn);
		ws.emit('muteFunction', soundOn);
		console.log(soundOn);
	}

	useEffect(() => {
		if (ws) {
			ws.on();
		}
	}, [ws]);
	
	function SoundIcon() {
		if (soundOn){
			return(<VolumeUp className={"w-9 h-9"} />);
		} else {
			return(<MuteIcon className={"w-9 h-9 mx-auto"} />);
		}}
	
	return(
		<div className="flex w-min rounded-md h-min m-2 center-items justify-center">
			<IconButton onClick={handleClick} aria-label="mute" variant="contained" color="error" size="small">
				<SoundIcon />
			</IconButton>
		</div>
	);
}

export default MuteButton;