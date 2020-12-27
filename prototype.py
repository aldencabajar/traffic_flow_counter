## a stand-in text file to provide an empty data/ directory
import numpy as np
import cv2
import os

config =  'darknet/cfg/yolov3.cfg'
wt_file = 'data/yolov3.weights'

# read darknet model for yolov3
net = cv2.dnn.readNetFromDarknet(config, wt_file)

# get layer names
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

[i for i in net.getUnconnectedOutLayers()]

# load image from data
img = cv2.imread('data/2018-01-04_metro_transportation-investment-innovation.jpg')
print(img.shape)

layerOutputs = net.forward(ln)



