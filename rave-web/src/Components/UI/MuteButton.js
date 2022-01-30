import { MuteIcon } from "../../Ressources/icons";
import IconButton from "@mui/material/Button";
import React, { useContext, useEffect, useState } from "react";

function MuteButton() {
	return(
		<div className="flex w-min rounded-md h-min m-2 center-items justify-center">
			<IconButton aria-label="mute" variant="contained" color="error" size="small">
				<MuteIcon className={"w-9 h-9 mx-auto"} />
			</IconButton>
		</div>
	);
}

export default MuteButton;