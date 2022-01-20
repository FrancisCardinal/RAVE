import Button from "@mui/material/Button";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import { useState } from "react";
import { Link } from "react-router-dom";
import {MenuIcon} from "../../Ressources/icons";

function NavMenu({ className }) {
	const [anchorEl, setAnchorEl] = useState(null);
	const open = Boolean(anchorEl);
  const handleClick = (event) => {
		setAnchorEl(event.currentTarget);
	};
	const handleClose = () => {
		setAnchorEl(null);
	};

	return (
    <div className={className}>
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
				<MenuItem onClick={handleClose}><Link to={`/`}>Home</Link></MenuItem>
				<MenuItem onClick={handleClose}><Link to={`/settings`}>Settings</Link></MenuItem>
				<MenuItem onClick={handleClose}><Link to={`/help`}>Help</Link></MenuItem>
			</Menu>
    </div>
  );
}

export default NavMenu;