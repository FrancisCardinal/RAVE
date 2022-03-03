const common = {
  homePage: {
    title: 'Assisted Listening',
    eyeTrackingLabel: "Eye Tracking Mode",
    faces: 'Faces :',
    forceRefresh: 'Refresh'
  },
  settingsPage: {
    title: 'Settings',
    language: {
      label: 'Language',
      french: 'French',
      english: 'English',
    },
    connectionLabel: 'Connection Status:',
    visionCalibration: 'Audio-Visual Calibration',
    eyeTrackerCalibration: 'Eye-Tracker Calibration',
  },
  helpPage: {
    title: 'Help',
    instruction:{
      label: 'Instructions',
      step1: "Create a configuration for the eye tracking mode or select your last configuration from the Settings page.",
      step2: "Put on the glasses and the headphones.",
      step3: "Verify the device's connection via the Settings page.",
      step4: "From the Home page, select the person you wish to listen by tapping on the person's face on your screen.",
      },
    },
  navigationBar:{
    homePage: "Home",
    settingsPage: "Settings",
    helpPage: 'Help',
  },
  visionCalibrationPage: {
    title: 'Visual-Audio Calibration',
    pointsLabel: 'Nb of points',
    orderLabel: 'Order of polynomial',
    confirm: 'Set',
    next: 'Next',
  },
  eyeTrackerCalibrationPage: {
    title: 'Eye-Traker Calibration',
    placeholder: "Choose a calibration configuration",
    next: 'Next',
    instruction: 'Replicate the mouvement shown bellow and press "Next" for the following mouvement:',
    modalTitle: 'Save new configuration',
    configName: 'Name',
    configPlaceholder: 'Enter your full name',
    errorMessage: 'The name you have chosen alreay exists, choose another one.',
  }
};
export default common;