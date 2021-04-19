dependencies: add_darknet get_tiny_yolo_v3_weights get_yolo_v3_weights
all_dcker: get_tiny_yolo_v3_weights get_yolo_v3_weights

add_darknet:
	git submodule init 
	git submodule update 

get_tiny_yolo_v3_weights:
	wget https://pjreddie.com/media/files/yolov3-tiny.weights
	mv yolov3-tiny.weights data/

get_yolo_v3_weights:
	wget https://pjreddie.com/media/files/yolov3.weights
	mv yolov3.weights data/



