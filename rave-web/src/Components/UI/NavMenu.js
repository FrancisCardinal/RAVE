import Button from "@mui/material/Button";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";
import {MenuIcon} from "../../Ressources/icons";
import PropTypes from 'prop-types';

NavMenu.propTypes = {
  className: PropTypes.string,
};
function NavMenu(props) {
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
    <div className={props.className}>
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
  );
}

export default NavMenu;