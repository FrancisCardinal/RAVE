import RAVE from '../../Ressources/RAVE.png';
import React, { useState } from "react";
import Button from "@mui/material/Button";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";
import {MenuIcon} from "../../Ressources/icons";

/**
 * This component displays the available pages of the website when the menu button is pressed
 * if the website is opened on a mobile device.
 */
export function MobileMainBar() {
  const [t] = useTranslation('common');
	const [anchorEl, setAnchorEl] = useState(null);
	const open = Boolean(anchorEl);

  const handleClick = (event) => {
		setAnchorEl(event.currentTarget);
	};

	const handleClose = () => {
		setAnchorEl(null);
	};

  return (
    <div className="flex flex-row bg-grey text-black font-mono h-14 inset-x-0 top-0 items-center">
      <img className='w-14 h-full' src={RAVE} alt="raves logo"/>
      <div className="absolute px-2 right-0">
        <Button
          id="menu-button"
          color="error"
          aria-controls={open ? 'menu-button' : undefined}
          aria-haspopup="true"
          aria-expanded={open ? 'true' : undefined}
          onClick={handleClick}
        >
          <MenuIcon className={"w-12 h-12 mx-auto"}/>
        </Button>
        <Menu
          id="nav-menu"
          anchorEl={anchorEl}
          open={open}
          onClose={handleClose}
          MenuListProps={{
            'aria-labelledby': 'menu-button',
            }}
        >
          <MenuItem onClick={handleClose}><Link to={`/`}>{t('navigationBar.homePage')}</Link></MenuItem>
          <MenuItem onClick={handleClose}><Link to={`/settings`}>{t('navigationBar.settingsPage')}</Link></MenuItem>
          <MenuItem onClick={handleClose}><Link to={`/help`}>{t('navigationBar.helpPage')}</Link></MenuItem>
        </Menu>
      </div>
    </div>
  );
}

export default MobileMainBar;