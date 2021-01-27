import { 
  Streamlit,
  StreamlitComponentBase,
  ComponentProps,
  withStreamlitConnection
} from "streamlit-component-lib";
import React, {useEffect, useState } from "react";
import { Slider } from 'baseui/slider';
import { ThemeProvider, styled} from "baseui";
import { useStyletron } from "styletron-react";

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
  const [css] = useStyletron()

  useEffect(() => Streamlit.setFrameHeight())



  return (
    <>
      <p>{label}</p>
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
        overrides= {{
          Thumb: {
            style: ({ $theme }) => ({
              backgroundColor: $theme.colors.negative400,
              height: '14px',
              width: '14px'
              
            })
          },
          InnerTrack: {
              style: ({  $theme, $isDragged}) => ({
                height: "2px",

              })
            },
          Root: { style: ({ $theme }) => ({
              backgroundColor: $theme.colors.primary50,
              positive600: $theme.colors.negative400
            })
          },
          Track: {
            style: ({ $theme }) => ({
              height: "4px", 
            })
          

            }

        }}
      />
    </>
  )
}

export default withStreamlitConnection(CustomSlider)