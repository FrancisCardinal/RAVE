const common = {
  homePage: {
    title: 'Écoute assistée',
    eyeTrackingLabel: "Mode suivi de l'oeil",
    faces: 'Visages :',
    forceRefresh: 'Actualiser'
  },
  settingsPage: {
    title: 'Réglages',
    language: {
      label: 'Langue',
      french: 'Français',
      english: 'Anglais',
    },
    connectionLabel: 'État de la connexion:',
    visionCalibration: 'Calibration audio-visuelle',
    eyeTrackerCalibration: 'Calibration eye-tracker',
    debugMode: 'Mode debuggage',
  },
  helpPage: {
    title: 'Aide',
    instruction:{
      label: 'Instructions',
      step1: "Aller créer votre configuration pour le mode suivi de l'oeil ou sélectionner votre configuration à partir des réglages.",
      step2: "Mettre les lunettes et le casque d'écoute.",
      step3: "Verifier la connexion avec l'appareil à partir des réglages.",
      step4: "À partir de la page d'acceuil, sélectionner la personne que vous désirez écouter en appuyant sur leur visage dans la zone de visages détectés.",
      },
    },
  navigationBar:{
    homePage: "Accueil",
    settingsPage: "Réglages",
    helpPage: 'Aide',
  },
  visionCalibrationPage: {
    title: 'Calibration audio-visuelle',
    pointsLabel: 'Nb de points',
    orderLabel: 'Ordre du polynomial',
    confirm: 'OK',
    next: 'Prochain'
  },
  eyeTrackerCalibrationPage: {
    title: "Calibration du suivi de l'oeil",
    placeholder: "Choisir une configuration",
    next: 'Prochain',
    instruction: 'Faire le mouvement indiqué et appuyer sur "Prochain":',
    modalTitle: 'Sauvegarder la configuration',
    configName: 'Nom',
    configPlaceholder: 'Entrer votre nom complet',
    errorMessage: 'Le nom que vous avez choisi existe déjà, choisir un nouveau.',
  }
};
export default common;