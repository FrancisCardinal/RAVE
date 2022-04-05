import React, { createContext, useState, FC } from 'react';
interface IDebbugingContext {
  debugging: boolean;
  toggleDebugging: () => void;
}
const defaultState = {
  debugging: false,
  toggleDebugging: () => {console.log("default")},
};

export const DebugContext = createContext<IDebbugingContext>(defaultState);

const DebugContextProvider : FC = ({children}) => {
  const [debugging, setDebugging] = useState(false);

  const toggleDebugging = () => {
    setDebugging(!debugging);
  };
  return (
    <DebugContext.Provider
      value={{
        debugging,
        toggleDebugging
      }}>
        {children}
    </DebugContext.Provider>
  );
}

export default DebugContextProvider;