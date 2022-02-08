import React, { useContext } from "react";
import SocketContext from "../../socketContext";
import { useTranslation } from 'react-i18next';

function CalibButton() {
  const [t] = useTranslation("common");
	const ws = useContext(SocketContext);

  return (
    <div className="">
      <button
              className="px-4 py-2 font-semibold text-sm bg-grey text-black rounded-md shadow-sm"
              onClick={() => {
                ws.emit('nextCalibTarget');
              }}
            >
              {t('calibrationPage.next')}
      </button>
    </div>
  );
}

export default CalibButton;