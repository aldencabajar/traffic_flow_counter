import os
import streamlit.components.v1 as components

_component_func = components.declare_component(
    "custom_slider",
    url="http://localhost:3001",
)
_RELEASE = False 

#def custom_slider(label: str, minVal: int, maxVal: int,
#enabled: bool, value: int = 0, key = None) ->int:
#    component_value = _component_func(label =label, minVal = minVal, maxVal = maxVal, 
#    initialValue = [value], key=key, default = [value], enabled= enabled)
#    return component_value[0]

def custom_slider(label: str, minVal: int, maxVal: int, enabled: bool,  
                    InitialValue: int = 0, key = None):
    return(_component_func(label = label, minVal = minVal, maxVal = maxVal,
                              enabled = enabled, InitialValue= InitialValue, key = key))


# Add some test code to play with the component while it's in development.
# During development, we can run this just as we would any other Streamlit
# app: `$ streamlit run my_component/__init__.py`
if not _RELEASE:
    import streamlit as st

    st.subheader("Test components")

    # Create an instance of our component with a constant `name` arg, and
    # print its output value.
    with st.sidebar:
        st.header("Parameters")
        val = custom_slider(label = "Model Confidence", minVal = 0, maxVal = 100, 
        InitialValue = 70, enabled = True, key=2)
        print(val)
        custom_slider(label = "Overlap threshold", minVal = 0, maxVal = 100,
         InitialValue = 50, enabled= True, key=1)


