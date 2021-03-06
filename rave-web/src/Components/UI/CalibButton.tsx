import React from 'react';
import { useTranslation } from 'react-i18next';
import { useEmit } from '../../Hooks';
import { NextCalibTargetEvent } from 'rave-protocol/pythonEvents';

/**
 * This component is a button that is clicked when
 * the user wants to move to the next target in the visual-audio calibration.
 */
function CalibButton() {
  const [t] = useTranslation('common');
  const emit = useEmit();

  return (
    <div>
      <button
        className="px-4 py-2 font-semibold text-sm bg-grey text-black rounded-md shadow-sm"
        onClick={() => {
          emit(NextCalibTargetEvent());
        }}
      >
        {t('visionCalibrationPage.next')}
      </button>
    </div>
  );
}

export default CalibButton;
