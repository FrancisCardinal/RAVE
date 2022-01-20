import LanguageSelection from "../../Components/UI/LanguageSelection";

function SettingsScreen() {
  return (
    <>
      <h1 className="text-3xl font-bold underline">Réglages</h1>
      <LanguageSelection className={"p-4 max-w-md"}/>
    </>
  );
}

export default SettingsScreen;
