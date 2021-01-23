def loop_over_frames(frame, frame_counter, start): 
    # if frame is read correctly ret is True
    #if not ret:
    #    print("Can't receive frame (stream end?). Exiting ...")
    #    break


    labels, current_boxes, confidences = obj_detector.ForwardPassOutput(frame)
    frame = tc.drawBoxes(frame, labels, current_boxes, confidences) 
    new_car_count = tracker.TrackCars(current_boxes)

    frame_counter += 1
    frm = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return(frm, new_car_count, frame_counter)



