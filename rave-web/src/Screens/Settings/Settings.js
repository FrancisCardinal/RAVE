import { useTranslation } from "react-i18next";
import LanguageSelection from "../../Components/UI/LanguageSelection";

function SettingsScreen() {
  const [t, i18n] = useTranslation('common');
  return (
    <>
      <h1 className="text-3xl font-bold underline">{t('settingsPage.title')}</h1>
      <LanguageSelection className={"p-4 max-w-md"}/>
    </>
  );
}

export default SettingsScreen;
