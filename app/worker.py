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



if __name__ == '__main__':
    with Connection(conn):
        config =  'app/yolov3.cfg'
        wt_file = 'app/yolov3.weights'
        tracker = tc.CarsInFrameTracker(num_previous_frames = 10, frame_shape = (720, 1080))
        obj_detector = tc.ObjectDetector(wt_file, config, confidence = 0.7, nms_threshold=0.5)

        worker = Worker(map(Queue, listen))
        worker.work()