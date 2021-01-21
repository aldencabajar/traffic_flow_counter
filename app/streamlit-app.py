import streamlit as st
import numpy
import sys
import os
import tempfile
sys.path.append(os.getcwd())
import traffic_counter as tc
import cv2 
import time
import SessionState
from random import randint
from streamlit import caching
import copy


config =  'darknet/cfg/yolov3.cfg'
wt_file = 'data/yolov3.weights'
# set network
tracker = tc.CarsInFrameTracker(num_previous_frames = 10, frame_shape = (720, 1080))
obj_detector = tc.ObjectDetector(wt_file, config, confidence = 0.7, nms_threshold=0.5)

def main():

    state = SessionState.get(upload_key = None)
    caching.clear_cache()

    hide_streamlit_widgets()
    st.markdown('# Vehicle Counter') 
    st.markdown('Upload a video file to track and count vehicles.')
    sidebar_options()

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


        # add start and stop buttons to interupt processing
        if start:
            start_button.empty()
            stop = stop_button.button("stop")
            start = False
            # close out temp files
            f.close()
            tfile.close()
            #update upload key
            state.upload_key = str(randint(1000, int(1e6)))
            print(state.upload_key)
            loop_over_frames(vf, stop)

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

def loop_over_frames(vf, stop): 

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

    new_car_count_txt = st.empty()
    fps_meas_txt = st.empty()
    bar = st.progress(frame_counter)
    stframe = st.empty()
    start = time.time()

    while vf.isOpened():
        if stop:
            break
        ret, frame = vf.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
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



def sidebar_options():
    st.sidebar.markdown('## Parameters')
    st.sidebar.slider('Model Confidence')

main()





