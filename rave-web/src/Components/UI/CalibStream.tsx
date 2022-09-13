import React, { useState } from 'react';
import { useEventListener } from '../../Hooks';
import { CLIENT_EVENTS } from 'rave-protocol/clientEvents';

/**
 * This component displays the video stream of the vision camera for the visual-audio calibration.
 */
function CalibStream() {
  const [frame, setFrame] = useState({});

  useEventListener(CLIENT_EVENTS.CALIBRATION_FRAME, ({ frame: newFrame, dimensions : _dimensions }) => {
    setFrame(newFrame);
  });

  return (
    <div className="mx-4 mt-2">
      <img src={'data:image/jpeg;base64,' + frame} alt="calib video" />
    </div>
  );
}

export default CalibStream;
