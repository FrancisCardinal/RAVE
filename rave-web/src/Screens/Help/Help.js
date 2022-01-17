import LanguageSelection from "../../Components/UI/LanguageSelection";

function HelpScreen() {
  return (
    <div className="flex flex-col items-center">
      <h1 className="text-3xl font-bold underline">HelpScreen</h1>

      <div className="flex flex-col w-3/4 bg-grey p-6 m-4 rounded-xl shadow-md">
        <h2 className="text-2xl pb-2">Instructions</h2>
        <ol className="pl-5 space-y-3 list-decimal marker:font-bold text-justify">
          <li className="hover:font-bold">Mettre les lunettes et le casque d'écoute.</li>
          <li className="hover:font-bold">Verifier la connexion avec l'appareil.</li>
          <li className="hover:font-bold">À partir de la page Home, sélectionner la 
          personne que vous désirez écouter en appuyant sur leur visage dans la zone de visages détectés.</li>
        </ol>
      </div>
      <div className="bg-grey"><LanguageSelection /></div>
    </div>
  );
}

export default HelpScreen;
