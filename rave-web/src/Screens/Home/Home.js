import EyeTrackingMode from "../../Components/UI/EyeTrackingMode";
import MuteButton from "../../Components/UI/MuteButton";
import VolumeSlider from "../../Components/UI/VolumeSlider";

function HomeScreen() {
  return (
    <div className=" flex flex-col max-h-full items-center">
      <div className="p-1"><h1 className="text-3xl font-bold underline text-center">Écoute assistée</h1></div>
      <div className="flex flex-col relative w-min items-center bottom-0">
        <EyeTrackingMode />
        <MuteButton />
        <VolumeSlider />
      </div>
    </div>
  );
}

export default HomeScreen;
