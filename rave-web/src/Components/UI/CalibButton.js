import Button from "@mui/material/Button";
import React, { useEffect, useContext } from "react";
import SocketContext from "../../socketContext";

function CalibButton() {
  
	const ws = useContext(SocketContext);

  return (
    <div className="m-2">
      <button
              className="px-4 py-2 font-semibold text-sm bg-grey text-black rounded-md shadow-sm"
              onClick={() => {
                ws.emit('nextCalibTarget');
              }}
            >
              Next
            </button>
      {/* <Button variant="contained" color="error" onClick={handleClick}>Next</Button> */}
    </div>
  );
}

export default CalibButton;