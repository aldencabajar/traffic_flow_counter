import numpy as np
import cv2
import os
import time
import palettable

config =  'darknet/cfg/yolov3.cfg'
wt_file = 'data/yolov3.weights'

# set confidence param
confidence = 0.5
threshold = 0.3

# read darknet model for yolov3
net = cv2.dnn.readNetFromDarknet(config, wt_file)


# get layer names
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# load labels from COCO dataset
lbl_path = 'darknet/data/coco.names'
LABELS = open(lbl_path).read().strip().split('\n')

# load image from data
img = cv2.imread('data/2018-01-04_metro_transportation-investment-innovation.jpg')
(H, W) = img.shape[:2]
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

start = time.time()
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

end = time.time()


lbls = [LABELS[i] for i in class_lst]
print(lbls)

print('total processing time is ', end - start, 'seconds')

#apply non maximum suppression which outputs the final predictions 
idx = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold).flatten()

# drawing bounding boxes to the original image and corresponding confidence score
thickness = 3 
colors = palettable.cartocolors.qualitative.Bold_5.colors
unique_labels = list(set([lbls[i] for i in idx]))
color_lbl_dict = {unique_labels[i] : colors[i] for i in range(len(unique_labels))}

for i in idx:
    start_coord = tuple(boxes[i][:2])
    w, h = boxes[i][2:]
    end_coord = start_coord[0] + w, start_coord[1] + h
    conf = confidences[i]
    color = color_lbl_dict[lbls[i]] 

# text to be included to the output image
    txt = '{} ({})'.format(lbls[i], round(conf,3))
    img = cv2.rectangle(img, start_coord, end_coord, color, thickness)
    img = cv2.putText(img, txt, start_coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        

# write the output to a new image
cv2.imwrite('tmp_img_w_bb.jpg', img)

