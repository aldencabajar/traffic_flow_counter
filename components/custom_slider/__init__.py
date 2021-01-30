import os
import streamlit.components.v1 as components

root_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(root_dir, 'frontend/build')

# Toggle between dev and production mode
_RELEASE = True

if _RELEASE:
    _component_func = components.declare_component(
        "custom_slider",
        path = build_dir
    )


else:
    _component_func = components.declare_component(
        "custom_slider",
        url="http://localhost:3001",
    )


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

    with st.sidebar:
        st.header("Parameters")
        val = custom_slider(label = "Model Confidence", minVal = 0, maxVal = 100, 
        InitialValue = 70, enabled = True, key=2)
        print(val)
        custom_slider(label = "Overlap threshold", minVal = 0, maxVal = 100,
         InitialValue = 50, enabled= False, key=1)


