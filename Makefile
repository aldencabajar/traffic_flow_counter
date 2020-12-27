add_darknet:
	git submodule init 
	git submodule update 

get_tiny_yolo_v3_weights:
	wget https://pjreddie.com/media/files/yolov3-tiny.weights

get_yolo_v3_weights:
	wget https://pjreddie.com/media/files/yolov3.weights
