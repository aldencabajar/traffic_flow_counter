import os
import redis
from rq import Worker, Queue, Connection
import traffic_counter as tc
import numpy as np
import cv2
import time

listen = ['high', 'default', 'low']
redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost:6379')
conn = redis.from_url(redis_url)

config =  'darknet/cfg/yolov3.cfg'
wt_file = 'data/yolov3.weights'
tracker = tc.CarsInFrameTracker(num_previous_frames = 10, frame_shape = (720, 1080))
obj_detector = tc.ObjectDetector(wt_file, config, confidence = 0.7, nms_threshold=0.5)


def loop_over_frames(frame, frame_counter, start): 
    # if frame is read correctly ret is True
    #if not ret:
    #    print("Can't receive frame (stream end?). Exiting ...")
    #    break

    labels, current_boxes, confidences = obj_detector.ForwardPassOutput(frame)
    frame = tc.drawBoxes(frame, labels, current_boxes, confidences) 
    new_car_count = tracker.TrackCars(current_boxes)

    frame_counter += 1
    frm = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return(frm, new_car_count, frame_counter)




if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(map(Queue, listen))
        worker.work()