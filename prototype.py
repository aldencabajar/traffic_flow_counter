## a stand-in text file to provide an empty data/ directory
import numpy as np
import cv2
import os

config =  'darknet/cfg/yolov3.cfg'
wt_file = 'data/yolov3.weights'

# set confidence param
confidence = 0.5

# read darknet model for yolov3
net = cv2.dnn.readNetFromDarknet(config, wt_file)

# get layer names
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

[i for i in net.getUnconnectedOutLayers()]

# load labels from COCO dataset
lbl_path = 'darknet/data/coco.names'
LABELS = open(lbl_path).read().strip().split('\n')

# load image from data
img = cv2.imread('data/2018-01-04_metro_transportation-investment-innovation.jpg')
(W, H) = img.shape[:2]
print(img.shape)

# create a blob as input to the model
blob = cv2.dnn.blobFromImage(img, 1/255., (416, 416), swapRB=True, crop = False)
net.setInput(blob)
print(img.shape)

layerOutputs = net.forward(ln)
[i.shape for i in layerOutputs]


# initialize lists for the class, width and height 
# and x,y coords for bounding box

class_lst = []
boxes = []
confidences = []

for output in layerOutputs:
    for detection in output:
        # do not consider the first five values as these correspond to 
        # the x-y coords of the center, width and height of the bounding box,
        # and the objectness score
        scores = detection[5:]

        # get the index with the max score
        class_id = np.argmax(scores)
        conf = scores[class_id]

        if conf >= confidence:
            # scale the predictions back to the original size of e
            box = detection[0:4] * np.array([W,H]*2) 
            (cX, cY, width, height) = box.astype(int)

            # get the top and left-most coordinate of the bounding box
            x = int(cX - (width / 2))
            y = int(cY - (height / 2))

            #update list
            boxes.append([x, y, width, height])
            class_lst.append(class_id)
            confidences.append(conf)



lbls = [LABELS[i] for i in class_lst]
print(lbls)


        


