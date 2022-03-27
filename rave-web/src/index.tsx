import "./index.css";
import React from "react";
import ReactDOM from "react-dom";
import App from "./App";
import i18next from "i18next";
import { I18nextProvider } from "react-i18next";
import common_en from "./translations/en/common";
import common_fr from "./translations/fr/common";

i18next.init({
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: {
    en: {
      common: common_en
    },
    fr: {
      common: common_fr
    }
  },
});

ReactDOM.render(
  <React.StrictMode>
    <I18nextProvider i18n={i18next}>
      <App />
    </I18nextProvider>
  </React.StrictMode>,
  document.getElementById("root")
);
