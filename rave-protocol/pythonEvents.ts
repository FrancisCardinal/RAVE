import { AbstractMessage, MESSAGE_DESTINATIONS } from "./abstractMessages";

interface AbstractPythonMessage extends AbstractMessage {
  event: PYTHON_EVENTS;
  destination : MESSAGE_DESTINATIONS.PYTHON;
}

export enum PYTHON_EVENTS {
  ACTIVATE_EYE_TRACKING = 'activateEyeTracking',
  CHANGE_VISION_CALIBRATION_PARAMS = 'changeCalibParams',
  CHANGE_VISION_MODE = 'changeVisionMode',
  DELETE_CONFIG = 'deleteEyeTrackingCalib',
  END_EYE_TRACKER_CALIBRATION = 'endEyeTrackingCalib',
  EYE_TRACKER_ADD_NEW_CONFIG = 'addEyeTrackingCalib',
  EYE_TRACKING_CONFIG_SELECTED = 'selectEyeTrackingCalib',
  EYE_TRACKER_RESUME_CALIBRATION = 'resumeEyeTrackingCalib',
  EYE_TRACKER_PAUSE_CALIBRATION =  'pauseEyeTrackingCalib',
  FORCE_REFRESH = 'forceRefresh',
  GO_TO_VISION_CALIBRATION = 'goToVisionCalibration',
  GO_TO_EYE_TRACKER_CALIBRATION = 'goToEyeTrackingCalibration',
  MUTE_REQUEST = 'muteRequest',
  NEXT_CALIBRATION_TARGET = 'nextCalibTarget',
  QUIT_CALIBRATION = 'quitVisionCalibration',
  SET_OFFSET_EYE_TRACKER_CALIBRATION = 'setOffsetEyeTrackingCalib',
  SET_VOLUME = 'setVolume',
  START_EYE_TRACKER_CALIBRATION = 'startEyeTrackingCalibration',
  TARGET_SELECT = 'targetSelect',
  GET_TARGET = 'getTarget',
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

export function GetTargetEvent() {
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.GET_TARGET,
  }
}

export function NextCalibTargetEvent() {
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.NEXT_CALIBRATION_TARGET,
  }
};

export function DeleteConfigEvent(id : string) {
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.DELETE_CONFIG,
    payload: {
      id
    }
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

export function EyeTrackerResumeCalibEvent() {
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.EYE_TRACKER_RESUME_CALIBRATION,
  }
};

export function EyeTrackerPauseCalibEvent() {
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.EYE_TRACKER_PAUSE_CALIBRATION,
  }
};

export function QuitCalibrationEvent() {
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.QUIT_CALIBRATION,
  }
};

interface ChangeCalibrationParamsEvent {
  number : number;
  order : number;
}

export function ChangeVisionCalibrationParamsEvent(numberOfPoints : number, orderPolynomial : number) {
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.CHANGE_VISION_CALIBRATION_PARAMS,
    payload : {
      number : numberOfPoints,
      order : orderPolynomial,
    } as ChangeCalibrationParamsEvent
  }
};

export function ChangeVisionModeEvent(visionMode: string) {
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.CHANGE_VISION_MODE,
    payload : {
      mode : visionMode,
    }
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
  }
}

export function GoToVisionCalibrationEvent(){
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.GO_TO_VISION_CALIBRATION,
  }
}

export function ForceRefreshEvent(){
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.FORCE_REFRESH,
  }
}

export function StartEyeTrackerCalibrationEvent(){
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.START_EYE_TRACKER_CALIBRATION,
  }
}

export function EndEyeTrackerCalibrationEvent(){
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.END_EYE_TRACKER_CALIBRATION,
  }
}

export function SetOffsetEyeTrackerCalibrationEvent(){
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.SET_OFFSET_EYE_TRACKER_CALIBRATION,
  }
}

export function EyeTrackerAddNewConfigEvent(configName : string){
  return {
    destination : MESSAGE_DESTINATIONS.PYTHON,
    event: PYTHON_EVENTS.EYE_TRACKER_ADD_NEW_CONFIG,
    payload : {
      configName
    }
  }
}