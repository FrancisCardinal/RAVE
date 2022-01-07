import { BrowserRouter, Routes, Route } from "react-router-dom";

import HomeScreen from "./Screens/Home";
import SettingsScreen from "./Screens/Settings";
import HelpScreen from "./Screens/Help";
import { Navbar } from "./Components/UI";
function App() {
  return (
    <>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<HomeScreen />} />
          <Route path="/settings" element={<SettingsScreen />} />
          <Route path="/help" element={<HelpScreen />} />
        </Routes>
        <Navbar Color={"black"} />
      </BrowserRouter>
    </>
  );
}

export default App;
