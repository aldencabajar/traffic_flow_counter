import numpy as np
import cv2
import os
import time
import sys
import palettable
import tqdm

# set parameters 
confidence = 0.5
threshold = 0.3
config =  'darknet/cfg/yolov3.cfg'
wt_file = 'data/yolov3.weights'
video_file ='data/4K Road traffic video for object detection and tracking - free download now!.mp4'  
out_video_file = 'annotated.avi'

FRAME_RATE = 5 # there is really no need of getting each and every frame

# load labels from COCO dataset
lbl_path = 'darknet/data/coco.names'
LABELS = open(lbl_path).read().strip().split('\n')

# read darknet model for yolov3
net = cv2.dnn.readNetFromDarknet(config, wt_file)

# get layer names
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def ForwardPassOutput(frame, threshold = 0.5):
    # create a blob as input to the model
    H, W = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255., (416, 416), swapRB=True, crop = False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # initialize lists for the class, width and height 
    # and x,y coords for bounding box

    class_lst = []
    boxes = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            # do not consider the frst five values as these correspond to 
            # the x-y coords of the center, width and height of the bounding box,
            # and the objectness score
            scores = detection[5:]

            # get the index with the max score
            class_id = np.argmax(scores)
            conf = scores[class_id]

            if conf >= confidence:
                # scale the predictions back to the original size of image
                box = detection[0:4] * np.array([W,H]*2) 
                (cX, cY, width, height) = box.astype(int)

                # get the top and left-most coordinate of the bounding box
                x = int(cX - (width / 2))
                y = int(cY - (height / 2))

                #update list
                boxes.append([int(i) for i in [x, y, width, height]])
                class_lst.append(class_id)
                confidences.append(float(conf))
    #apply non maximum suppression which outputs the final predictions 
    idx = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold).flatten()
    return [LABELS[class_lst[i]] for i in idx], [boxes[i] for i in idx], [confidences[i] for i in idx]

def drawBoxes(frame, labels, boxes, confidences):
    boxColor = (128, 255, 0) # very light green
    TextColor = (255, 255, 255) # white
    boxThickness = 3 
    textThickness = 2

    for lbl, box, conf in zip(labels, boxes, confidences):
        start_coord = tuple(box[:2])
        w, h = box[2:]
        end_coord = start_coord[0] + w, start_coord[1] + h

    # text to be included to the output image
        txt = '{} ({})'.format(lbl, round(conf,3))
        frame = cv2.rectangle(frame, start_coord, end_coord, boxColor, boxThickness)
        frame = cv2.putText(frame, txt, start_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, TextColor, 2)

    return frame

def DetermineBoxCenter(box): 
    cx = int(box[0] + (box[2]/2))
    cy = int(box[1] + (box[3]/2))

    return [cx, cy]

def IsCarOnEdge(box, frame_shape = None, percent_win_edge = 10):
    # consider vertical dimension for now
    start_x, start_y = tuple(box[:2])
    W, H = box[2:]

    CarOnEdge = (start_y + H) >= (frame_shape[1] * (1 - percent_win_edge)/100) 

    return CarOnEdge




def TrackCarsInFrame(current_boxes, frame_shape = None,  prev_boxes_dict = None):
    if prev_boxes_dict is None:
        box_dict = {}
        # since these are fresh car instances, add id numbers from 0 to n cars 
        ids = np.arange(len(current_boxes))
        for box, id in zip(current_boxes, ids):
            box_dict[id] = {'box': box, 'center': DetermineBoxCenter(box)}
        return box_dict

    # create an array of box centers from previous box dicts
    maxID = max(list(prev_boxes_dict.keys()))
    curr_boxes_dict = dict(prev_boxes_dict)
    prev_box_centers = np.array(
        [v['center'] for k, v in prev_boxes_dict.items()])
    current_box_centers = np.array(
        [DetermineBoxCenter(box) for box in current_boxes])

    dist = np.linalg.norm(
        prev_box_centers[:, None, :] - current_box_centers[None, :, :], 
        axis = 2)
    #get the index with the minimum distance
    min_idx = np.argmin(dist, axis = 1)

    # get the corresponding distances
    shp = min_idx.shape[0]
    ind = np.arange(1, shp + 1) * shp - (shp - min_idx) 
    min_dist =  np.take(dist, ind)

    # if there are indices in the current frame with no matches from the previous, 
    # assign new id and add to box dictionary
    idx_new_centers = np.setdiff1d(
        np.arange(current_box_centers.shape[0]), np.unique(min_idx))
    new_centers = current_box_centers[idx_new_centers] 

    for i, idx in enumerate(idx_new_centers):
        curr_boxes_dict[maxID + i + 1] = {'box': current_boxes[idx], 
        'center': list(new_centers[i])}

    
    # remove centers that have boxes that are within 10% of video frame edge 
    #CarsOnEdge =     
    return curr_boxes_dict 



def getFrames(video_file, num_frames = None):
    cap = cv2.VideoCapture(video_file)
    out_list = []

    for i in range(num_frames):
        _ , frame = cap.read()
        labels, boxes, confidences = ForwardPassOutput(frame) 
        out_list.append({
            'boxes' : boxes,
            'frame' : frame
        })

    return out_list

frames_tmp = getFrames(video_file, 2)
lbls = [', '.join([str(j) for  j in i[:2]]) for i in frames_tmp[1]['boxes']]
img = drawBoxes(frames_tmp[1]['frame'], lbls , frames_tmp[1]['boxes'], [0.1]*8)
img2 = drawBoxes(frames_tmp[0]['frame'], lbls , frames_tmp[0]['boxes'], [0.1]*8)
cv2.imwrite('tmp.jpg', img)

prev_box_dict= TrackCarsInFrame(frames_tmp[0]['boxes'])
tmp = TrackCarsInFrame(frames_tmp[1]['boxes'], prev_box_dict)

max(list(prev_box_dict.keys()))

##### Main program #######
# Initialize video stream 
cap = cv2.VideoCapture(video_file)

try:
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) 
    print('Total number of frames to be processed:', num_frames,
    '\nFrame rate (frames per second):', fps)
except:
    print('We cannot determine number of frames and FPS!')

grab = True 
counter = 0
num_frames_processed = 500 
writer = None

start = time.time()
pbar = tqdm.tqdm(total = 100)

while grab & (counter < 500):
    (grab, frame) = cap.read()
    H, W = frame.shape[:2]
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(out_video_file, fourcc, fps,  (W, H), True) 

    if (((counter + 1) % int(fps // FRAME_RATE)) == 0) or (counter == 0):  
        labels, boxes, confidences = ForwardPassOutput(frame) 
        frame = drawBoxes(frame, labels, boxes, confidences) 
    counter += 1
    pbar.update(100/num_frames_processed)
    writer.write(frame)

writer.release()
pbar.close()
cap.release()

end = time.time()

print('total processing time =', round(end - start, 3), 's')


