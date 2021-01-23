import numpy as np
import cv2
import os
import time
import sys
import palettable
import tqdm
import copy

# load labels from COCO dataset
try:
    LABELS = open('app/coco.names').read().strip().split('\n')
except:
    LABELS = open('darknet/data/coco.names').read().strip().split('\n')

class ObjectDetector:
    def __init__(self, weights, config, confidence, nms_threshold):
        self.net = cv2.dnn.readNetFromDarknet(config, weights)
        self.confidence = confidence
        self.nms_threshold = nms_threshold 

        # identify layers
        ln = self.net.getLayerNames() 
        self.ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def ForwardPassOutput(self, frame):
        # create a blob as input to the model
        H, W = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255., (416, 416), swapRB=True, crop = False)

        # predict using model
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)

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

                if conf >= self.confidence:
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
        idx = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.nms_threshold).flatten()
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
    ImmediatePrevious = copy.deepcopy(previous_frames[-1])

    unmatched_idx = np.arange(len(current_boxes))
    unmatchedIds = []
    for frame in previous_frames:
        unmatchedIds += list(frame.keys())
    unmatchedIds = list(set(unmatchedIds))
        

    for frame in reversed(previous_frames):

        if len(unmatched_idx) > 0 and len(unmatchedIds) > 0:
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

            tmp_current_boxes = [current_boxes[i] for i in unmatched_idx] 
            current_box_centers = np.array(
            [DetermineBoxCenter(box) for box in tmp_current_boxes])

            # if there are no retrieved centers from the frame, skip over it since 
            # it means that that frame does not contain the id
            if len(prev_box_centers) == 0:
                continue
            
            # get the corresponding distances
            dist = GetDistBetweenCenters(prev_box_centers, current_box_centers)
            #get the index with the minimum distance
            min_idx = np.argmin(dist, axis = 1)
            ind = GetFlattenedIndex(min_idx, dist.shape)
            min_dist =  np.take(dist, ind)

            # detect indices with distances greater than max of 
            # the two dimensions
            max_dim = np.array([max(v['box'][2:]) for k, v in frame.items() if k in unmatchedIds])  / 2
            grt_than = [min_dist[i] > j for i, j in enumerate(max_dim)]
            
            
            # update values in current box dict 
            for  i in np.where(np.logical_not(grt_than))[0]:
                id_ = tmp_ids[i]
                box = tmp_current_boxes[min_idx[i]]
                center = list(current_box_centers[min_idx[i]])
                if id_ not in list(ImmediatePrevious.keys()):
                    ImmediatePrevious[id_] ={'box': box, 'center': center}
                else:
                    ImmediatePrevious[tmp_ids[i]]['box'] = box
                    ImmediatePrevious[tmp_ids[i]]['center'] = center

            unmatchedIds = list(np.setdiff1d(np.array(unmatchedIds), 
            tmp_ids[np.where(np.logical_not(grt_than))[0]])
            )
            unmatched_idx = np.setdiff1d(unmatched_idx, 
            unmatched_idx[np.unique(min_idx[np.logical_not(grt_than)])])
    return unmatched_idx, ImmediatePrevious            


class CarsInFrameTracker:            

    def __init__(self, num_previous_frames, frame_shape):
        self.num_tracked_cars = 0 
        self.frames = []
        self.frame_shape = frame_shape
        self.num_previous_frames = num_previous_frames

        

    def TrackCars(self, current_boxes): 
        if len(self.frames) == 0:
            box_dict = {}
            # since these are fresh car instances, add id numbers from 0 to n cars 
            ids = np.arange(len(current_boxes))
            for box, id in zip(current_boxes, ids):
                box_dict[id] = {'box': box, 'center': DetermineBoxCenter(box)}
            self.num_tracked_cars = len(ids)
            self.frames.append(box_dict)

            return self.num_tracked_cars

        
        ImmediatePrevious = copy.deepcopy(self.frames[-1])

        ImmediatePreviousCenter = []
        max_dim = []

        for _, v in ImmediatePrevious.items():
            ImmediatePreviousCenter.append(v['center'])
            max_dim.append(max(v['box'][2:]))

        ImmediatePreviousCenter = np.array(ImmediatePreviousCenter)
        max_dim = np.array(max_dim) / 2
        unmatched_idx, curr_boxes_dict = CheckPreviousFrames(current_boxes, self.frames)
        
        # if there are indices in the current frame with no matches from the previous frames, 
        # assign new id and add to box dictionary
        new_centers = []

        if unmatched_idx.shape[0] > 0:  
            # check if these centers are real cars, by checking if the distance is greater than 
            # the maximum dimension
            for i in unmatched_idx:
                if not IsCarOnEdge(current_boxes[i], self.frame_shape, percent_win_edge = 20):
                    center = np.array([DetermineBoxCenter(current_boxes[i])])
                    dist = GetDistBetweenCenters(ImmediatePreviousCenter, center)
                    # are all identified cars sufficiently far from the "new center"?
                    isreal = np.all(np.greater(dist, max_dim))
                    if isreal:
                        new_centers.append((i, center))
            if len(new_centers) > 0:
                for i, (idx, nc) in enumerate(new_centers):
                    curr_boxes_dict[self.num_tracked_cars + i] = {'box': current_boxes[idx], 
                    'center': list(nc.flatten())}

 
        CarsOnEdge = []     
        for k, v in curr_boxes_dict.items():
            if IsCarOnEdge(v['box'], self.frame_shape, percent_win_edge = 20): 
                CarsOnEdge.append(k) 
            
        for key in CarsOnEdge:
            del curr_boxes_dict[key]
        
        
        num_new_cars = len(new_centers)
        self.num_tracked_cars += num_new_cars
        self.frames.append(curr_boxes_dict)

        if len(self.frames) > self.num_previous_frames:
            self.frames.pop(0)

        return self.num_tracked_cars


if __name__ == '__main__':
    ##### Main program for testing #######
    # Initialize video stream 
    # set parameters 
    config =  'darknet/cfg/yolov3.cfg'
    wt_file = 'data/yolov3.weights'
    video_file ='data/4K Road traffic video for object detection and tracking - free download now!.mp4'  
    out_video_file = 'annotated_try.avi'
    FRAME_RATE = 30 # there is really no need of getting each and every frame

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

    # Initialize objects
    tracker = CarsInFrameTracker(num_previous_frames = 10, frame_shape = (720, 1080))
    obj_detector = ObjectDetector(wt_file, config, confidence = 0.7, nms_threshold=0.5)


    while grab & (counter < num_frames_processed):
        (grab, frame) = cap.read()

        #print('iteration', counter + 1)
        pbar.set_postfix({'iteration': counter + 1,
        'number of cars': tracker.num_tracked_cars})
        H, W = frame.shape[:2]
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(out_video_file, fourcc, fps,  (W, H), True) 

        if (((counter + 1) % int(fps // FRAME_RATE)) == 0) or (counter == 0):  
            labels, current_boxes, confidences = obj_detector.ForwardPassOutput(frame)
            frame = drawBoxes(frame, labels, current_boxes, confidences) 
            new_car_count = tracker.TrackCars(current_boxes)

            #print('new car count = ', new_car_count)
            frame = cv2.putText(frame, '{} {}'.format('total car count:', str(new_car_count)), 
            (800, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frame = cv2.putText(frame, '{} {}'.format('current car count:', len(tracker.frames[-1])), 
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

