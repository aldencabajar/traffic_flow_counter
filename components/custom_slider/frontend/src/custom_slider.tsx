import { 
  Streamlit,
  StreamlitComponentBase,
  ComponentProps,
  withStreamlitConnection
} from "streamlit-component-lib";
import React, {useEffect, useState } from "react";
import { Slider } from 'baseui/slider';

interface pythonArgs {
  label: string
  minVal: number
  maxVal: number
  initialValue: number
  enabled: boolean


}

const CustomSlider = (props: ComponentProps) => {
  // Destructure using Typescript interface
  // This ensures typing validation for received props from Python
  const {label, minVal, maxVal, initialValue, enabled} : pythonArgs = props.args 
  const [value, setValue] = useState([initialValue])

  useEffect(() => Streamlit.setFrameHeight())

  return (
    <>
      {label}
      <Slider
        disabled={!enabled}
        value={value}
        onChange={({ value }) => value && setValue(value)}
        onFinalChange={({ value }) => {
          Streamlit.setComponentValue(value)
          console.log(value)
        }}
        min={minVal}
        max={maxVal}
      />
    </>
  )
}

export default withStreamlitConnection(CustomSlider)