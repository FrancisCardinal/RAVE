import { Link } from "react-router-dom";
import { HomeIcon, SettingsIcon, HelpIcon } from "../../Ressources/icons";

export function Navbar() {
  return (
    <>
      <nav className="fixed bottom-0 inset-x-0 bg-blue-100 flex justify-between text-sm text-blue-900 uppercase font-mono divide-x divide-black-500">
        <Link
          className="w-full block py-5 px-3 text-center hover:bg-blue-200"
          to={`/`}
        >
          <HomeIcon className={"w-6 h-6 mb-2 mx-auto"} />
          Home
        </Link>
        <Link
          className="w-full block py-5 px-3 text-center hover:bg-blue-200"
          to={`/settings`}
        >
          <SettingsIcon className={"w-6 h-6 mb-2 mx-auto"} />
          Settings
        </Link>
        <Link
          className="w-full block py-5 px-3 text-center hover:bg-blue-200"
          to={`/help`}
        >
          <HelpIcon className={"w-6 h-6 mb-2 mx-auto"} />
          Help
        </Link>
      </nav>
    </>
  );
}

export default Navbar;
