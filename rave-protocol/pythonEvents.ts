import { AbstractMessage, MESSAGE_DESTINATIONS } from "./abstractMessages";

interface AbstractPythonMessage extends AbstractMessage {
  event: PYTHON_EVENTS;
  destination : MESSAGE_DESTINATIONS.PYTHON;
}

export enum PYTHON_EVENTS {
  FORCE_REFRESH = 'forceRefresh',
  TARGET_SELECT = 'targetSelect',
  GO_TO_VISION_CALIBRATION = 'goToVisionCalibration',
  GO_TO_EYE_TRACKER_CALIBRATION = 'goToEyeTrackerCalibration',
  STOP_VISION_CALIBRATION = 'stopVisonCalibration',
  NEXT_CALIBRATION_TARGET = 'nextCalibTarget',
  MUTE_REQUEST = 'muteRequest',
  ACTIVATE_EYE_TRACKING = 'activateEyeTracking',
  SET_VOLUME = 'setVolume',
  CHANGE_CALIBRATION_PARAMS = 'changeCalibrationParams',
  DELETE_CONFIG = 'deleteConfig',
  EYE_TRACKING_CONFIG_SELECTED = 'eyeTrackingConfigSelected',
  QUIT_CALIBRATION = 'quitCalibration',
}

export interface TargetSelectPayload {
  targetId : number;
}

export function TargetSelectEvent(targetId : number) {
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.TARGET_SELECT,
    payload : {
      targetId
    } as TargetSelectPayload
  }
};

export function NextCalibTargetEvent() {
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.NEXT_CALIBRATION_TARGET,
    payload : {}
  }
};

export function DeleteConfigEvent() {
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.DELETE_CONFIG,
    payload : {}
  }
};

export interface EyeTrackingConfigSelectedEventPayload {
  name : string
}

export function EyeTrackingConfigSelectedEvent(name : string) {
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.EYE_TRACKING_CONFIG_SELECTED,
    payload : {
      name
    } as EyeTrackingConfigSelectedEventPayload
  }
};

export function QuitCalibrationEvent() {
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.QUIT_CALIBRATION,
    payload : {}
  }
};

interface ChangeCalibrationParamsEvent {
  number : number;
  order : number;
}

export function ChangeCalibrationParamsEvent(numberOfPoints : number, orderPolynomial : number) {
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.CHANGE_CALIBRATION_PARAMS,
    payload : {
      number : numberOfPoints,
      order : orderPolynomial,
    } as ChangeCalibrationParamsEvent
  }
};

export function ActivateEyeTrackingEvent(onStatus : boolean) {
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.ACTIVATE_EYE_TRACKING,
    payload : {
      onStatus
    }
  }
};

export function MuteRequestEvent(muteStatus : boolean) {
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.MUTE_REQUEST,
    payload : {
      muteStatus
    }
  }
};

export function SetVolumeEvent(volume : number) {
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.SET_VOLUME,
    payload : {
      volume
    }
  }
};

export function GoToEyeTrackerCalibrationEvent(){
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.GO_TO_EYE_TRACKER_CALIBRATION,
    payload : {}
  }
}

export function GoToVisionCalibrationEvent(){
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.GO_TO_VISION_CALIBRATION,
    payload : {}
  }
}

export function ForceRefreshEvent(){
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.FORCE_REFRESH,
    payload : {}
  }
}