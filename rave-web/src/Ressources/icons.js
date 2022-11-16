/* eslint-disable react/style-prop-object */
/*
  All icons are aquired freely through : https://materialdesignicons.com
 - Select an icon 
 - Click the code icon (</>)
 - View SVG
 - Create an icon component
 - Paste the SVG in the return function
 - Remove the style attribute and replace it with className
 - Export the component
*/
import React from "react";
import PropTypes from "prop-types";

HomeIcon.propTypes = {
  className: PropTypes.string,
}
export function HomeIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24">
      <path
        fill="currentColor"
        d="M10,20V14H14V20H19V12H22L12,3L2,12H5V20H10Z"
      />
    </svg>
  );
}
SettingsIcon.propTypes = {
  className: PropTypes.string,
}
export function SettingsIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24">
      <path
        fill="currentColor"
        d="M12,8A4,4 0 0,1 16,12A4,4 0 0,1 12,16A4,4 0 0,1 8,12A4,4 0 0,1 12,8M12,10A2,2 0 0,0 10,12A2,2 0 0,0 12,14A2,2 0 0,0 14,12A2,2 0 0,0 12,10M10,22C9.75,22 9.54,21.82 9.5,21.58L9.13,18.93C8.5,18.68 7.96,18.34 7.44,17.94L4.95,18.95C4.73,19.03 4.46,18.95 4.34,18.73L2.34,15.27C2.21,15.05 2.27,14.78 2.46,14.63L4.57,12.97L4.5,12L4.57,11L2.46,9.37C2.27,9.22 2.21,8.95 2.34,8.73L4.34,5.27C4.46,5.05 4.73,4.96 4.95,5.05L7.44,6.05C7.96,5.66 8.5,5.32 9.13,5.07L9.5,2.42C9.54,2.18 9.75,2 10,2H14C14.25,2 14.46,2.18 14.5,2.42L14.87,5.07C15.5,5.32 16.04,5.66 16.56,6.05L19.05,5.05C19.27,4.96 19.54,5.05 19.66,5.27L21.66,8.73C21.79,8.95 21.73,9.22 21.54,9.37L19.43,11L19.5,12L19.43,13L21.54,14.63C21.73,14.78 21.79,15.05 21.66,15.27L19.66,18.73C19.54,18.95 19.27,19.04 19.05,18.95L16.56,17.95C16.04,18.34 15.5,18.68 14.87,18.93L14.5,21.58C14.46,21.82 14.25,22 14,22H10M11.25,4L10.88,6.61C9.68,6.86 8.62,7.5 7.85,8.39L5.44,7.35L4.69,8.65L6.8,10.2C6.4,11.37 6.4,12.64 6.8,13.8L4.68,15.36L5.43,16.66L7.86,15.62C8.63,16.5 9.68,17.14 10.87,17.38L11.24,20H12.76L13.13,17.39C14.32,17.14 15.37,16.5 16.14,15.62L18.57,16.66L19.32,15.36L17.2,13.81C17.6,12.64 17.6,11.37 17.2,10.2L19.31,8.65L18.56,7.35L16.15,8.39C15.38,7.5 14.32,6.86 13.12,6.62L12.75,4H11.25Z"
      />
    </svg>
  );
}
HelpIcon.propTypes = {
  className: PropTypes.string,
}
export function HelpIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24">
      <path
        fill="currentColor"
        d="M10,19H13V22H10V19M12,2C17.35,2.22 19.68,7.62 16.5,11.67C15.67,12.67 14.33,13.33 13.67,14.17C13,15 13,16 13,17H10C10,15.33 10,13.92 10.67,12.92C11.33,11.92 12.67,11.33 13.5,10.67C15.92,8.43 15.32,5.26 12,5A3,3 0 0,0 9,8H6A6,6 0 0,1 12,2Z"
      />
    </svg>
  );
}
MuteIcon.propTypes = {
  className: PropTypes.string,
}
export function MuteIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24">
      <path 
        fill="currentColor" 
        d="M5.64,3.64L21.36,19.36L19.95,20.78L16,16.83V20L11,15H7V9H8.17L4.22,5.05L5.64,3.64M16,4V11.17L12.41,7.58L16,4Z" />
    </svg>
  );
}
VolumeDown.propTypes = {
  className: PropTypes.string,
}
export function VolumeDown({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24">
      <path 
        fill="currentColor" 
        d="M5,9V15H9L14,20V4L9,9M18.5,12C18.5,10.23 17.5,8.71 16,7.97V16C17.5,15.29 18.5,13.76 18.5,12Z" />
    </svg>
  );
}
VolumeUp.propTypes = {
  className: PropTypes.string,
} 
export function VolumeUp({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24">
      <path 
        fill="currentColor" 
        d="M14,3.23V5.29C16.89,6.15 19,8.83 19,12C19,15.17 16.89,17.84 14,18.7V20.77C18,19.86 21,16.28 21,12C21,7.72 18,4.14 14,3.23M16.5,12C16.5,10.23 15.5,8.71 14,7.97V16C15.5,15.29 16.5,13.76 16.5,12M3,9V15H7L12,20V4L7,9H3Z" />
    </svg>
  );
}
MenuIcon.propTypes = {
  className: PropTypes.string,
}
export function MenuIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24">
    <path 
      fill="currentColor" 
      d="M3,6H21V8H3V6M3,11H21V13H3V11M3,16H21V18H3V16Z" />
</svg>
  );
}
WifiCheckedIcon.propTypes = {
  className: PropTypes.string,
}
export function WifiCheckedIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24">
      <path 
        fill="green" 
        d="M12 12C9.97 12 8.1 12.67 6.6 13.8L4.8 11.4C6.81 9.89 9.3 9 12 9S17.19 9.89 19.2 11.4L17.92 13.1C17.55 13.17 17.18 13.27 16.84 13.41C15.44 12.5 13.78 12 12 12M21 9L22.8 6.6C19.79 4.34 16.05 3 12 3S4.21 4.34 1.2 6.6L3 9C5.5 7.12 8.62 6 12 6S18.5 7.12 21 9M12 15C10.65 15 9.4 15.45 8.4 16.2L12 21L13.04 19.61C13 19.41 13 19.21 13 19C13 17.66 13.44 16.43 14.19 15.43C13.5 15.16 12.77 15 12 15M17.75 19.43L16.16 17.84L15 19L17.75 22L22.5 17.25L21.34 15.84L17.75 19.43Z" />
    </svg>
  );
}
NoWifiConnectionIcon.propTypes = {
  className: PropTypes.string,
}
export function NoWifiConnectionIcon(props) {
  return (
    <svg className={props.className} viewBox="0 0 24 24">
    <path 
      fill="red" 
      d="M12 12C9.97 12 8.1 12.67 6.6 13.8L4.8 11.4C6.81 9.89 9.3 9 12 9S17.19 9.89 19.2 11.4L17.92 13.1C17.55 13.17 17.18 13.27 16.84 13.41C15.44 12.5 13.78 12 12 12M21 9L22.8 6.6C19.79 4.34 16.05 3 12 3S4.21 4.34 1.2 6.6L3 9C5.5 7.12 8.62 6 12 6S18.5 7.12 21 9M12 15C10.65 15 9.4 15.45 8.4 16.2L12 21L13.04 19.61C13 19.41 13 19.21 13 19C13 17.66 13.44 16.43 14.19 15.43C13.5 15.16 12.77 15 12 15M21.12 15.46L19 17.59L16.88 15.47L15.47 16.88L17.59 19L15.47 21.12L16.88 22.54L19 20.41L21.12 22.54L22.54 21.12L20.41 19L22.54 16.88L21.12 15.46Z" />
  </svg>
  );
}

ErrorIcon.propTypes = {
  className: PropTypes.string,
}
export function ErrorIcon(props) {
  return (
    <svg className={props.className} viewBox="0 0 24 24">
    <path fill="#D32F2F" d="M13,13H11V7H13M13,17H11V15H13M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2Z" />
    </svg>
  );
}

AddIcon.propTypes = {
  className: PropTypes.string,
}
export function AddIcon(props) {
  return (
    <svg className={props.className} viewBox="0 0 24 24">
      <path fill="currentColor" d="M19,13H13V19H11V13H5V11H11V5H13V11H19V13Z" />
    </svg>
  );
}

DeleteIcon.propTypes = {
  className: PropTypes.string,
}
export function DeleteIcon(props) {
  return (
    <svg className={props.className} viewBox="0 0 24 24">
      <path fill="currentColor" d="M19,4H15.5L14.5,3H9.5L8.5,4H5V6H19M6,19A2,2 0 0,0 8,21H16A2,2 0 0,0 18,19V7H6V19Z" />
    </svg>
  );
}

SaveIcon.propTypes = {
  className: PropTypes.string,
}
export function SaveIcon(props) {
  return (
    <svg className={props.className} viewBox="0 0 24 24">
    <path fill="currentColor" d="M15,9H5V5H15M12,19A3,3 0 0,1 9,16A3,3 0 0,1 12,13A3,3 0 0,1 15,16A3,3 0 0,1 12,19M17,3H5C3.89,3 3,3.9 3,5V19A2,2 0 0,0 5,21H19A2,2 0 0,0 21,19V7L17,3Z" />
</svg>
  );
}

YesIcon.propTypes = {
  className: PropTypes.string,
}
export function YesIcon(props) {
  return (
    <svg className={props.className} viewBox="0 0 24 24">
      <path fill="currentColor" d="M9,20.42L2.79,14.21L5.62,11.38L9,14.77L18.88,4.88L21.71,7.71L9,20.42Z" />
    </svg>
  );
}

NoIcon.propTypes = {
  className: PropTypes.string,
}
export function NoIcon(props) {
  return (
    <svg className={props.className} viewBox="0 0 24 24">
      <path fill="currentColor" d="M20 6.91L17.09 4L12 9.09L6.91 4L4 6.91L9.09 12L4 17.09L6.91 20L12 14.91L17.09 20L20 17.09L14.91 12L20 6.91Z" />
    </svg>
  );
}

PlayIcon.propTypes = {
  className: PropTypes.string,
}
export function PlayIcon(props) {
  return (
    <svg className={props.className} viewBox="0 0 24 24">
      <path fill="currentColor" d="M8,5.14V19.14L19,12.14L8,5.14Z" />
    </svg>
  );
}

StopIcon.propTypes = {
  className: PropTypes.string,
}
export function StopIcon(props) {
  return (
    <svg className={props.className} viewBox="0 0 24 24">
      <path fill="currentColor" d="M14,19H18V5H14M6,19H10V5H6V19Z" />
    </svg>
  );
}

CameraIcon.propTypes = {
  className: PropTypes.string,
}
export function CameraIcon(props) {
  return (
    <svg className={props.className} viewBox="0 0 24 24">
      <path fill="currentColor" d="M4,4H7L9,2H15L17,4H20A2,2 0 0,1 22,6V18A2,2 0 0,1 20,20H4A2,2 0 0,1 2,18V6A2,2 0 0,1 4,4M12,7A5,5 0 0,0 7,12A5,5 0 0,0 12,17A5,5 0 0,0 17,12A5,5 0 0,0 12,7M12,9A3,3 0 0,1 15,12A3,3 0 0,1 12,15A3,3 0 0,1 9,12A3,3 0 0,1 12,9Z" />
    </svg>
  );
}