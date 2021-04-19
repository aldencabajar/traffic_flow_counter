#!/bin/bash

# setup submodule
git submodule init
git submodule update 

# download necessary models 
wget https://pjreddie.com/media/files/yolov3.weights

mv yolov3.weights data/

