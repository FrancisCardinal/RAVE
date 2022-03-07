import { useTranslation } from "react-i18next";
import React from "react";

function HelpScreen() {
  const [t] = useTranslation('common');
  const instructions = [
    t('helpPage.instruction.step1'),
    t('helpPage.instruction.step2'),
    t('helpPage.instruction.step3'),
    t('helpPage.instruction.step4'),
  ]
  return (
    <div className="flex flex-col items-center">
      <h1 className="text-3xl font-bold underline">{t('helpPage.title')}</h1>

      <div className="flex flex-col max-w-2xl bg-grey p-6 m-4 rounded-xl shadow-md">
        <h2 className="text-2xl pb-2">{t('helpPage.instruction.label')}</h2>
        <ol className="pl-5 space-y-3 list-decimal marker:font-bold text-justify">
          {instructions.map((step) => <li key={Math.random()} className="hover:font-bold">{step}</li>)}
          
        </ol>
      </div>
    </div>
  );
}

export default HelpScreen;
