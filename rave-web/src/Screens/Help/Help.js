import { useTranslation } from "react-i18next";

function HelpScreen() {
  const [t, i18n] = useTranslation('common');
  return (
    <div className="flex flex-col items-center">
      <h1 className="text-3xl font-bold underline">{t('helpPage.title')}</h1>

      <div className="flex flex-col w-3/4 bg-grey p-6 m-4 rounded-xl shadow-md">
        <h2 className="text-2xl pb-2">{t('helpPage.instruction.label')}</h2>
        <ol className="pl-5 space-y-3 list-decimal marker:font-bold text-justify">
          <li className="hover:font-bold">{t('helpPage.instruction.step1')}</li>
          <li className="hover:font-bold">{t('helpPage.instruction.step2')}</li>
          <li className="hover:font-bold">{t('helpPage.instruction.step3')}</li>
        </ol>
      </div>
    </div>
  );
}

export default HelpScreen;
