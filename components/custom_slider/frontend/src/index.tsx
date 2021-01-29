import React from "react"
import ReactDOM from "react-dom"
import CustomSlider from "./custom_slider"

import { Client as Styletron } from "styletron-engine-atomic"
import { Provider as StyletronProvider } from "styletron-react"
import { ThemeProvider, LightTheme } from "baseui"

const engine = new Styletron()

// Wrap your CustomSlider with the baseui them
ReactDOM.render(
  <React.StrictMode>
    <CustomSlider />
  </React.StrictMode>,
  document.getElementById("root")
)