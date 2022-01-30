import EyeTrackingMode from '../../Components/UI/EyeTrackingMode';
import MuteButton from '../../Components/UI/MuteButton';
import VolumeSlider from '../../Components/UI/VolumeSlider';
import groupe from '../../Ressources/groupe.jpg';
import { BrowserView, MobileView } from 'react-device-detect';
import { useTranslation } from 'react-i18next';
import React from 'react';

function HomeScreen() {
  const [t] = useTranslation('common');
  return (
    <div className=" flex flex-col max-h-full items-center">
      <div className="p-1">
        <h1 className="text-3xl font-bold underline text-center">{t('homePage.title')}</h1>
      </div>
      <BrowserView className="flex w-full p-4 rounded-md items-center">
        <Stream />
      </BrowserView>

      <MobileView className="flex w-full mx-2 p-3 rounded-md items-center">
        <Stream />
      </MobileView>
      <div className="flex flex-col relative w-min items-center bottom-0">
        <EyeTrackingMode />
        <MuteButton />
        <VolumeSlider />
      </div>
    </div>
  );
}

export default HomeScreen;
