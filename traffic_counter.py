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
num_frames_processed = 50 
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

IsCarOnEdge(current_boxes[-1], frame.shape, 20)
total_car_count

tmp= TrackCarsInFrame(current_boxes, frame.shape, previous_box_dict) 
min_idx = np.argmin(tmp, axis = 1)
x, y = tmp.shape
ind = np.arange(y, (x+1)*y, step = y) - (y - min_idx)
np.take(tmp, ind)
tmp.flatten().shape


        