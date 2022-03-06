import { AbstractMessage, MESSAGE_DESTINATIONS } from "./abstractMessages";

interface AbstractClientMessage extends AbstractMessage {
  destination : MESSAGE_DESTINATIONS.CLIENT;
  event: CLIENT_EVENTS;
}

export enum CLIENT_EVENTS {
  CALIBRATION_FRAME = 'calibrationFrame',
  CALIBRATION_ERROR = 'calibrationError',
  CONNECTION_STATUS = 'connectionStatus',
  EYE_TRACKING_CONFIGURATIONS = 'configList',
  NEW_FRAME_AVAILABLE = 'newFrameAvailable',
};

export interface BoundingBox {
  id : number;
  dx : number;
  dy : number;
  width : number;
  height : number;
  color? : string;
}

export function CalibrationErrorEvent(message : string) {
  return {
    destination : MESSAGE_DESTINATIONS.CLIENT,
    event : CLIENT_EVENTS.CALIBRATION_ERROR,
    payload : { 
      message
    }
  }
};

export interface ConnectionStatusPayload {
  status : number;
}

export function ConnectionStatusEvent(status : number) {
  return {
    destination : MESSAGE_DESTINATIONS.CLIENT,
    event : CLIENT_EVENTS.CONNECTION_STATUS,
    payload : {
      status
    } as ConnectionStatusPayload
  }
}

export function EyeTrackingConfigurationsEvent(configuration : any) {
  return {
    destination : MESSAGE_DESTINATIONS.CLIENT,
    event : CLIENT_EVENTS.EYE_TRACKING_CONFIGURATIONS,
    payload : {
      configuration
    }
  }
}

export function CalibrationFrameEvent(frame : string, dimensions : [number, number, number]) {
  return {
    destination : MESSAGE_DESTINATIONS.CLIENT,
    event : CLIENT_EVENTS.CALIBRATION_FRAME,
    payload : {
      frame,
      dimensions,
    }
  }
}

export interface NewFrameAvailablePayload {
  /**
 * @param frame The base64 encoded text string
 * @param dimensions A tuple of the [width,height] parameters of the image 
 * @param boundingBoxes An array of boudingBoxes to be drawn on the image and their associated ID
 */
  base64Frame : string;
  dimensions : [number,number, number];
  boundingBoxes : BoundingBox[];
}

/**
 * @param base64Frame The base64 encoded text string
 * @param dimensions A tuple of the [width,height] parameters of the image 
 * @param boundingBoxes An array of boudingBoxes to be drawn on the image and their associated ID
 */
 export function NewFrameAvailableEvent(base64Frame : string, dimensions : [number, number, number], boundingBoxes : BoundingBox[]) {
  return {
    destination : MESSAGE_DESTINATIONS.CLIENT,
    event: CLIENT_EVENTS.NEW_FRAME_AVAILABLE,
    payload : {
      base64Frame,
      dimensions,
      boundingBoxes,
    } as NewFrameAvailablePayload
  }
};