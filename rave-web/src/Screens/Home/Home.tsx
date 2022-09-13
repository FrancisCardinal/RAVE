import React from 'react';
import EyeTrackingMode from '../../Components/UI/EyeTrackingMode';
import MuteButton from '../../Components/UI/MuteButton';
import VolumeSlider from '../../Components/UI/VolumeSlider';
import Stream from '../../Components/UI/Stream';
import { BrowserView, MobileView } from 'react-device-detect';
import { useTranslation } from 'react-i18next';
import { useEmit } from "../../Hooks";
import { ForceRefreshEvent } from 'rave-protocol/pythonEvents';

/**
 * This screen is the home screen. It displays the vision stream with the bounding boxes if faces are detected.
 * It also displays the volume controls and the eye-tracking mode switch.
 */
function HomeScreen() {
  const emit = useEmit();
  const [t] = useTranslation('common');
  return (
    <div className=" flex flex-col max-h-full items-center">
      <div className="p-1">
        <h1 className="text-3xl font-bold underline text-center">{t('homePage.title')}</h1>
      </div>
      <BrowserView className="flex w-3/4 p-4">
        <Stream />
      </BrowserView>

      <MobileView className="flex w-full">
        <Stream />
      </MobileView>
      <div className="flex flex-col relative w-min items-center bottom-0">
        <div className='flex flex-row'>
          <button
          className="flex mt-2 bg-grey h-min text-black rounded-lg shadow-sm"
          onClick={() => {
            emit(ForceRefreshEvent());
          }}
        >
          <h1 className='p-2 font-medium w-max'>{t('homePage.forceRefresh')}</h1>
        </button>
        <EyeTrackingMode />
        </div>
        
        <MuteButton />
        <VolumeSlider />
      </div>
    </div>
  );
}

export default HomeScreen;
