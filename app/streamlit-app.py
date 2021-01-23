import streamlit as st
import numpy
import sys
import os
import tempfile
sys.path.append(os.getcwd())
import traffic_counter as tc
import time
import utils.SessionState as SessionState
from random import randint
from streamlit import caching
import copy
from rq import Queue
from worker import conn, loop_over_frames
import cv2




q = Queue(connection = conn)

def main():

    state = SessionState.get(upload_key = None)
    caching.clear_cache()

    hide_streamlit_widgets()
    st.markdown('# Vehicle Counter') 
    st.markdown('Upload a video file to track and count vehicles.')
    st.sidebar.markdown('## Parameters')
    conf = st.sidebar.slider('Model Confidence', value = 70)
    nms = st.sidebar.slider('Non-Maximum Suppresion Threshold', value = 50)

    #set model confidence and nms threshold 
    #obj_detector.nms_threshold = nms / 100
    #obj_detector.confidence = conf / 100 


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

        # boolean to note end of file
        eof = False 

        # add start and stop buttons to interrupt processing
        if start:
            start_button.empty()
            stop = stop_button.button("stop")
            start = False
            # close out temp files
            tfile.close()
            #update upload key
            state.upload_key = str(randint(1000, int(1e6)))
            print(state.upload_key)
            frame_counter = 0
            new_car_count_txt = st.empty()
            fps_meas_txt = st.empty()
            #bar = st.progress(frame_counter)
            stframe = st.empty()
            start = time.time()

            while not eof:
                if stop:
                    break
                ret, frame = vf.read()
                job = q.enqueue(loop_over_frames, 
                                args =(frame, frame_counter, start),
                                result_ttl = 10 
                                )
                time.sleep(1.5)
                end = time.time()
                img, new_car_count, frame_counter = job.result
                fps_measurement = frame_counter/(end - start)

                #print(loop_over_frames(frame, frame_counter, start))
                new_car_count_txt.markdown(f'**Total car count:** {new_car_count}')
                fps_meas_txt.markdown(f'**Frames per second:** {fps_measurement:.2f}')
                stframe.image(img, width = 720)




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


main()





