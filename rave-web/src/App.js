import { BrowserRouter, Routes, Route } from "react-router-dom";

import HomeScreen from "./Screens/Home";
import SettingsScreen from "./Screens/Settings";
import HelpScreen from "./Screens/Help";
import TestRoom from "./Screens/Test";
import { Navbar } from "./Components/UI";

function App() {
  return (
    <>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<HomeScreen />} />
          <Route path="/settings" element={<SettingsScreen />} />
          <Route path="/help" element={<HelpScreen />} />
          <Route path="/serverTest" element={<TestRoom />} />
        </Routes>
        <Navbar />
      </BrowserRouter>
    </>
  );
}

export default App;
