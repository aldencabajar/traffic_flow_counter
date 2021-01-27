import streamlit as st
import numpy
import sys
import os
import tempfile
sys.path.append(os.getcwd())
import traffic_counter as tc
import cv2 
import time
import utils.SessionState as SessionState
from random import randint
from streamlit import caching
import copy
from components.custom_slider import custom_slider


config =  'darknet/cfg/yolov3.cfg'
wt_file = 'data/yolov3.weights'
# set network
#tracker = tc.CarsInFrameTracker(num_previous_frames = 10, frame_shape = (720, 1080))
#obj_detector = tc.ObjectDetector(wt_file, config, confidence = 0.7, nms_threshold=0.5)

@st.cache(
    hash_funcs={
        st.delta_generator.DeltaGenerator: lambda x: None,
        "_regex.Pattern": lambda x: None,
    },
    allow_output_mutation=True,
)
def load_obj_detector(config, wt_file):
    obj_detector = tc.ObjectDetector(wt_file, config, confidence = 0.7, nms_threshold=0.5)

    return obj_detector
    

def parameter_sliders(key, enabled = True):
    conf = custom_slider("Model Confidence", 
                        minVal = 0, maxVal = 100, value= 70, enabled = enabled,
                        key = key[0])
    nms = custom_slider('Non-Maximum Suppresion Threshold', 
                        minVal = 0, maxVal = 100, value= 50, enabled = enabled,
                        key = key[1])

        
    return(conf, nms)


def main():
    keys = ['conf', 'nms']

    obj_detector = load_obj_detector(config, wt_file)
    tracker = tc.CarsInFrameTracker(num_previous_frames = 10, frame_shape = (720, 1080))

    state = SessionState.get(upload_key = None, enabled = True, start = False)
    hide_streamlit_widgets()
    st.markdown('# Vehicle Counter') 
    st.markdown('Upload a video file to track and count vehicles. Don\'t forget to change parameters to tune the model!')
    with st.sidebar:
        st.markdown('## Parameters')
        conf, nms = parameter_sliders(keys, state.enabled)
        st.slider(label = "test")


    #set model confidence and nms threshold 
    obj_detector.nms_threshold = nms / 100
    obj_detector.confidence = conf / 100 
    print(obj_detector.nms_threshold)
    print(obj_detector.confidence)

    upload = st.empty()
    start_button = st.empty()
    stop_button = st.empty()

    with upload:
        f = st.file_uploader('Upload Video file (mpeg format)', key = state.upload_key)
    if f is not None:
        tfile  = tempfile.NamedTemporaryFile(delete = True)
        tfile.write(f.read())

        upload.empty()
        vf = cv2.VideoCapture(tfile.name)
        start = start_button.button("start")
        state.enabled = False


            


        # add start and stop buttons to interupt processing
        if start:
            start_button.empty()
            tfile.close()
            f.close()
            state.upload_key = str(randint(1000, int(1e6)))
            state.enabled = True 
            ProcessFrames(vf, tracker, obj_detector)


            


def hide_streamlit_widgets():
    """
    hides widgets that are displayed by streamlit when running
    """
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def ProcessFrames(vf, tracker, obj_detector): 

    # Main loop to process every frame and predict cars
    # get video attributes 
    try:
        num_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(vf.get(cv2.CAP_PROP_FPS)) 
        print('Total number of frames to be processed:', num_frames,
        '\nFrame rate (frames per second):', fps)
    except:
        print('We cannot determine number of frames and FPS!')


    frame_counter = 0
    stop = stop_button.button("stop")
    new_car_count_txt = st.empty()
    fps_meas_txt = st.empty()
    bar = st.progress(frame_counter)
    stframe = st.empty()
    start = time.time()

    while vf.isOpened():
        ret, frame = vf.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
        labels, current_boxes, confidences = obj_detector.ForwardPassOutput(frame)
        frame = tc.drawBoxes(frame, labels, current_boxes, confidences) 
        new_car_count = tracker.TrackCars(current_boxes)
        new_car_count_txt.markdown(f'**Total car count:** {new_car_count}')

        end = time.time()

        frame_counter += 1
        fps_measurement = frame_counter/(end - start)
        fps_meas_txt.markdown(f'**Frames per second:** {fps_measurement:.2f}')
        bar.progress(frame_counter/num_frames)

        frm = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frm, width = 720)

main()





