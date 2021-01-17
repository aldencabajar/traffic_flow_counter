import streamlit as st
import numpy
import sys
import os
import tempfile
sys.path.append(os.getcwd())
import traffic_counter as tc
import cv2 
import time

config =  'darknet/cfg/yolov3.cfg'
wt_file = 'data/yolov3.weights'
# set network
tracker = tc.CarsInFrameTracker(num_previous_frames = 10, frame_shape = (720, 1080))
obj_detector = tc.ObjectDetector(wt_file, config, confidence = 0.7, nms_threshold=0.5)

def main():
    st.markdown('# Vehicle Counter') 
    st.markdown('Upload a video file to track and count vehicles.')
    sidebar_options()

    upload = st.empty()
    upload.beta_expander(label = '', expanded = True)

    with upload:
        f = st.file_uploader('Upload Video file (mpeg format)')




    if f is not None:
        tfile  = tempfile.NamedTemporaryFile(delete = False)
        tfile.write(f.read())
        upload.empty()
        vf = cv2.VideoCapture(tfile.name)


        loop_over_frames(vf)

        



def loop_over_frames(vf): 

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
    st.sidebar.markdown('Model Confidence')

if __name__ == '__main__':
    main()





