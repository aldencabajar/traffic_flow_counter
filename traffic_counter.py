import numpy as np
import cv2
import os
import time
import sys
import palettable
import tqdm
import copy
import forward_yolo_net

# set parameters 
confidence = 0.8
threshold = 0.3
config =  'darknet/cfg/yolov3.cfg'
wt_file = 'data/yolov3.weights'
video_file ='data/4K Road traffic video for object detection and tracking - free download now!.mp4'  
out_video_file = 'annotated.avi'

FRAME_RATE = 30 # there is really no need of getting each and every frame

# load labels from COCO dataset
lbl_path = 'darknet/data/coco.names'
LABELS = open(lbl_path).read().strip().split('\n')

# read darknet model for yolov3
net = cv2.dnn.readNetFromDarknet(config, wt_file)

# get layer names
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def ForwardPassOutput(frame, net, layers, confidence_threshold, labels,  nms_threshold = 0.5):
    # create a blob as input to the model
    H, W = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255., (416, 416), swapRB=True, crop = False)
    net.setInput(blob)
    layerOutputs = net.forward(layers)

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

            if conf >= confidence_threshold:
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
    idx = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold).flatten()
    return [labels[class_lst[i]] for i in idx], [boxes[i] for i in idx], [confidences[i] for i in idx]

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
        txt = '{} ({})'.format(', '.join([str(i) for i in DetermineBoxCenter(box)]), round(conf,3))
        frame = cv2.rectangle(frame, start_coord, end_coord, boxColor, boxThickness)
        frame = cv2.putText(frame, txt, start_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, TextColor, 2)

    return frame

def DetermineBoxCenter(box):
    cx = int(box[0] + (box[2]/2))
    cy = int(box[1] + (box[3]/2))

    return [cx, cy]

def IsCarOnEdge(box, frame_shape = None, percent_win_edge = 10):
    # consider vertical dimension for now
    centers =  DetermineBoxCenter(box)
    CarOnEdge = centers[1] >= (frame_shape[0] * (1 - percent_win_edge/100)) 

    return CarOnEdge

def GetFlattenedIndex(rowwise_idx, shape_of_matrix):
    x, y = shape_of_matrix
    ind = np.arange(y, (x+1)*y, step = y) - (y - rowwise_idx)
    return ind


def GetDistBetweenCenters(box_center1, box_center2): 
    dist = np.linalg.norm(
        box_center1[:, None, :] - box_center2[None, :, :], 
        axis = 2)

    return dist


def CheckPreviousFrames(current_boxes, previous_frames):
    ### PROTOTYPE CODE FOR CHECKING PREVIOUS FRAMES 
    current_box_centers = np.array(
    [DetermineBoxCenter(box) for box in current_boxes])
    ImmediatePrevious = dict(previous_frames[-1])

    unmatchedIds = list(ImmediatePrevious.keys())
    unmatched_idx = np.arange(current_box_centers.shape[0])

    for frame in reversed(box_dicts[-11:-1]):

        if (len(unmatched_idx) > 0) or (len(unmatchedIds) > 0):
            tmp_ids = np.array([k for k, v in frame.items() if k in unmatchedIds])
            prev_box_centers = np.array(
            [v['center'] for k, v in frame.items() if k in unmatchedIds])

            prev_box_centers = []
            tmp_ids = []

            for k, v in frame.items():
                if k in unmatchedIds:
                    prev_box_centers.append(v['center'])
                    tmp_ids.append(k)
            tmp_ids = np.array(tmp_ids)
            prev_box_centers= np.array(prev_box_centers)

            # get the corresponding distances
            dist = GetDistBetweenCenters(prev_box_centers, current_box_centers[unmatched_idx])
            #get the index with the minimum distance
            min_idx = np.argmin(dist, axis = 1)
            ind = GetFlattenedIndex(min_idx, dist.shape)
            min_dist =  np.take(dist, ind)

            # detect indices with distances greater than half of the max of 
            # the two dimensions
            max_dim = np.array([max(v['box'][2:]) for k, v in frame.items() if k in unmatchedIds]) / 2
            grt_than = [min_dist[i] > j/2 for i, j in enumerate(max_dim)]
            unmatchedIds = list(tmp_ids[np.where(grt_than)[0]])
            unmatched_idx = np.in1d(unmatched_idx, np.unique(min_idx), invert = True)

            print(unmatchedIds, unmatched_idx, dist)
            


def TrackCarsInFrame(current_boxes, frame_shape = None, prev_boxes_dict = None):
    if prev_boxes_dict is None:
        box_dict = {}
        # since these are fresh car instances, add id numbers from 0 to n cars 
        ids = np.arange(len(current_boxes))
        for box, id in zip(current_boxes, ids):
            box_dict[id] = {'box': box, 'center': DetermineBoxCenter(box)}
        return box_dict, len(ids)

    # create an array of box centers from previous box dicts
    ids = list(prev_boxes_dict.keys())
    maxID = max(ids)
    curr_boxes_dict = copy.deepcopy(prev_boxes_dict)
    prev_box_centers = np.array(
        [v['center'] for k, v in prev_boxes_dict.items()])
    current_box_centers = np.array(
        [DetermineBoxCenter(box) for box in current_boxes])

    # get the corresponding distances
    dist = GetDistBetweenCenters(prev_box_centers, current_box_centers)

    #return dist, [v['box'][-1] for _, v in prev_boxes_dict.items()]


    #get the index with the minimum distance
    min_idx = np.argmin(dist, axis = 1)
    ind = GetFlattenedIndex(min_idx, dist.shape) 
    min_dist =  np.take(dist, ind)
    idxs, counts = np.unique(min_idx, return_counts = True)
    collisions = np.in1d(min_idx, idxs[np.where(counts > 1)[0]])


    # detect indices with distances greater than max of 
    # the two dimensions
    max_dim = [max(v['box'][2:]) for k, v in prev_boxes_dict.items()]
    grt_than = [min_dist[i] > j/2 for i, j in enumerate(max_dim)]
    ands = np.logical_not(np.logical_and(collisions, grt_than))

    #return collisions, GreaterThanMaxDim, ands
    for i, (cond_met, grt_than_cnd) in enumerate(zip(ands, grt_than)):
        if cond_met and not grt_than_cnd:
            curr_boxes_dict[ids[i]]['box'] = current_boxes[min_idx[i]]  
            curr_boxes_dict[ids[i]]['center'] = current_box_centers[min_idx[i]]


    # if there are indices in the current frame with no matches from the previous, 
    # assign new id and add to box dictionary
    IsRealCar = None
    idx_new_centers = np.setdiff1d(
        np.arange(current_box_centers.shape[0]), np.unique(min_idx))
    
    if idx_new_centers.shape[0] > 0:  
        # check if these centers are real cars, by checking if the distance is less than 
        # the maximum dimension
        IsRealCar = []
        for i in idx_new_centers:
            dist_tmp = dist[:, i] 
            isreal = np.all(np.greater(dist_tmp, np.array(max_dim)))
            IsRealCar.append(isreal)
        new_centers = current_box_centers[idx_new_centers[IsRealCar], :] 
        if np.sum(IsRealCar) > 0:
            for i, idx in enumerate(idx_new_centers):
                curr_boxes_dict[maxID + i + 1] = {'box': current_boxes[idx], 
                'center': list(new_centers[i])}

    print(curr_boxes_dict)
    # remove centers that have boxes that are within 10% of video frame edge 
    CarsOnEdge = []     
    for k, v in curr_boxes_dict.items():
        if IsCarOnEdge(v['box'], frame_shape, 35): 
           CarsOnEdge.append(k) 
           
    print(CarsOnEdge)
    print(current_box_centers)
    for key in CarsOnEdge:
        del curr_boxes_dict[key]
    
    num_new_cars = 0 if IsRealCar is None else np.sum(IsRealCar)

    return curr_boxes_dict, num_new_cars
    


frames_tmp = getFrames(video_file,18 , 19)
_, current_boxes, __ = ForwardPassOutput(frames_tmp[1]['frame'])
_, previous_boxes, __ = ForwardPassOutput(frames_tmp[0]['frame'])
dist, height = TrackCarsInFrame(current_boxes, prev_boxes_dict = box_dicts[18], frame_shape=(720, 1080))
tmp = TrackCarsInFrame(current_boxes, prev_boxes_dict = box_dicts[18], frame_shape=(720, 1080))
min_idx = np.argmin(dist, axis = 1)
idx = GetFlattenedIndex(min_idx, dist.shape)
np.greater(np.take(dist, idx), np.array(height)/2)

np.take(dist, idx)

len(box_dicts[20])
[DetermineBoxCenter(i) for i in current_boxes]



lbls = [', '.join([str(j) for  j in DetermineBoxCenter(i)]) for i in frames_tmp[0]['boxes']]
img = drawBoxes(frames_tmp[1]['frame'], lbls , frames_tmp[1]['boxes'], [0.1]*8)
img2 = drawBoxes(frames_tmp[0]['frame'], lbls , frames_tmp[0]['boxes'], [0.1]*8)
cv2.imwrite('tmp.jpg', img2)

cv2.imwrite('tmp2.jpg', img)

prev_box_dict, new_car_count = TrackCarsInFrame(frames_tmp[0]['boxes'])
tmp, new_car_count = TrackCarsInFrame(frames_tmp[1]['boxes'],
                        frame_shape = frames_tmp[1]['frame'].shape,
                         prev_boxes_dict= prev_box_dict)

for k, v in tmp[0].items():
    print(IsCarOnEdge(v['box']))


print(prev_box_dict)
print(tmp)

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
num_frames_processed = 10 
writer = None

start = time.time()
pbar = tqdm.tqdm(total = 100)
previous_box_dict = None
total_car_count = 0

box_dicts = []

while grab & (counter < num_frames_processed):
    (grab, frame) = cap.read()

    print('iteration', counter + 1)
    H, W = frame.shape[:2]
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(out_video_file, fourcc, fps,  (W, H), True) 

    if (((counter + 1) % int(fps // FRAME_RATE)) == 0) or (counter == 0):  
        labels, current_boxes, confidences = ForwardPassOutput(frame) 
        frame = drawBoxes(frame, labels, current_boxes, confidences) 
        previous_box_dict, new_car_count = TrackCarsInFrame(current_boxes, frame.shape, previous_box_dict) 
        box_dicts.append(copy.deepcopy(previous_box_dict))

        total_car_count += new_car_count
        print('new car count = ', new_car_count)
        frame = cv2.putText(frame, '{} {}'.format('total car count:', str(total_car_count)), 
        (800, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        frame = cv2.putText(frame, '{} {}'.format('current car count:', len(previous_box_dict)), 
        (800, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    counter += 1
    # add frame number 
    frame = cv2.putText(frame, 'frame: ' + str(counter), (800, 110),
    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2 )
    pbar.update(100/num_frames_processed)
    writer.write(frame)

writer.release()
pbar.close()
cap.release()

end = time.time()

print('total processing time =', round(end - start, 3), 's')

