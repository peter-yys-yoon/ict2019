from tmp import *

class VehicleClass:
    def __init__(self):
        self.car_trajectory_dict = {}
        self.car_next_id = 0
        
        self.car_list_list = []
        
    def car_tracking(self, car_np, img_id):
        """
        Assign track_id to detected object.
        car_np: car objects in numpy form.
        """
        
        if car_np is None:
            return []

        new_car_bboxs = car_np[:, 1:5].astype(np.uint32)  # b/  x y w h c / cls_conf, cls_idx
        new_car_score = car_np[:, 5]
        cls_conf = car_np[:, 6]
        car_dest_list = []

        if img_id > 1:  # First frame does not have previous frame
            car_bbox_list_prev_frame = self.car_list_list[img_id - 1].copy()
        else:
            car_bbox_list_prev_frame = []

        # print('car bbox list prev frame ', len(car_bbox_list_prev_frame))
        for c, score, conf in zip(new_car_bboxs, new_car_score, cls_conf):
            car_bbox_det = c
            bbox_in_xywh = enlarge_bbox(car_bbox_det, enlarge_scale)
            bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)

            if img_id == 0:  # First frame, all ids are assigned automatically
                car_track_id = self.car_next_id
                self.car_next_id += 1
            else:
                car_track_id, match_index = get_track_id_SpatialConsistency(bbox_det,
                                                                            car_bbox_list_prev_frame)
                # print(car_track_id, match_index)
                if car_track_id != -1:  # if candidate from prev frame matched, prevent it from matching another
                    del car_bbox_list_prev_frame[match_index]

            bbox_det_dict = {"img_id": img_id,
                             "track_id": car_track_id,
                             "bbox": bbox_det,
                             "score": score,
                             "conf": conf}
            car_dest_list.append(bbox_det_dict)

        for car_bbox_det_dict in car_dest_list:  # detections for current frame
            if car_bbox_det_dict["track_id"] == -1:  # this id means matching not found yet
                car_bbox_det_dict["track_id"] = self.car_next_id
                self.car_next_id += 1
                
                
        self.car_list_list.append(car_dest_list)
        
        return car_dest_list
        
    def car_trajectory(self, det_list):
        for det in det_list:
            track_id = det['track_id']
            xyxy = xywh_to_x1y1x2y2(det['bbox'])

            if track_id in self.car_trajectory_dict.keys():
                track_obj = self.car_trajectory_dict[track_id]
                tracklet = track_obj[CAR_TRACKLET]
                history = track_obj[MOVE_HISTORY]
                tracklet.append(xyxy)
                
                if len(tracklet) > THRESHOLD_CAR_TRACKLET_SIZE: # 25
                    tracklet = tracklet[1:]
                    
                if len(tracklet) > 1:
                    dist, diag  = avg_dist(tracklet)
                    th = diag * 0.02   
                
                    if dist > th : # moved
                        history.append(1)
                    else: 
                        history.append(0)

                if len(history) > THRESHOLD_CAR_HISTORY_SIZE: # 120
                    history = history[1:]

                track_obj[CAR_TRACKLET] = tracklet
                track_obj[MOVE_HISTORY] = history 

            else:
                track_obj = {CAR_TRACKLET: [xyxy],
                                MOVE_HISTORY: [0],
                                GTA_HISTORY: [0]}

            self.car_trajectory_dict[track_id] = track_obj


    def parking_detection_sub(self, car_dest_list, img):
        
        stop_car, park_car = None, None
        
        for car in car_dest_list:
            x, y, w, h = car['bbox']
            track_id = car['track_id']
            score = car['score']
            conf = car['conf']

        
            tracker = self.car_trajectory_dict[track_id]
            history = tracker[MOVE_HISTORY]
            moving_active = np.sum(history[LAST_MOVED_MOVE_COUNT:])
            moving_stop = np.sum(history[LAST_MOVED_STOP_COUNT:])
            moving_parking = np.sum(history[LAST_MOVED_PARKING_COUNT:])

            cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_INACTIVE, 1)
            text_filled(img, (x, y), f'{track_id} Inactive', COLOR_INACTIVE)
            
            if moving_active: # moving now
                cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_MOVING, 1)
                text_filled(img, (x, y), f'CAR {track_id} Active', COLOR_MOVING)
                
            else: # not moving.

                if moving_stop : # moved last 5 sec
                    cv2.rectangle(img, (x, y), (x + w, y + h), YELLOW, 1)
                    text_filled(img, (x, y), f'CAR {track_id} STOP', YELLOW)

                    cropped = img[y:y + h, x:x + w, :]
                    mask = np.zeros(cropped.shape, dtype=img.dtype)
                    mask[:, :, :] = YELLOW
                    overlayed = cv2.addWeighted(cropped, 0.8, mask, 0.2, 0)
                    img[y:y + h, x:x + w, :] = overlayed[:, :, :]                    
                    stop_car = car
                    
                       
                elif moving_parking:
                    
                    cv2.rectangle(img, (x, y), (x + w, y + h), RED, 1)
                    text_filled(img, (x, y), f'CAR {track_id} PARKING', RED)

                    cropped = img[y:y + h, x:x + w, :]
                    mask = np.zeros(cropped.shape, dtype=img.dtype)
                    
                    mask[:, :, :] = RED
                    masked = cv2.addWeighted(cropped, 0.8, mask, 0.2, 0)
                    img[y:y + h, x:x + w, :] = masked[:, :, :]
                    park_car = car

        return stop_car, park_car
    

    def parking_detection(self, car_dest_list, img, img_id):


        
        parking_flag = 0
        warning_flag = 0
        imgW, imgH = img.shape[1], img.shape[0]
        
        stop_car, park_car = self.parking_detection_sub(car_dest_list, img)
                
        HEADER_height = int(img.shape[1] * 0.05)
        mask = np.zeros((HEADER_height, imgW, 3), dtype=np.uint8)
        
        if stop_car:
            mask[:,:,:] = YELLOW
            msg = f'Frame: {str(img_id).rjust(4)}, Vehicle stopped'
            car = stop_car
        elif park_car:
            mask[:,:,:] = RED
            msg = f'Frame: {str(img_id).rjust(4)}, Redzone parking!'
            car = park_car
        else:
            mask[:,:,:] = BALCK
            msg = f'Frame: {str(img_id).rjust(4)}'
            car = None

        if car:
            x, y, w, h = car['bbox']
            cx, cy = int((x*2+w)/2), int((y*2+h)/2)
            overlayfunc(img, (cx,cy), max(w,h)) 
        header = cv2.addWeighted(img[0:HEADER_height, 0:imgW,:], 0.4, mask, 0.6, 0)
        
        txt_size, baseLine1 = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 2, 2)
        p1_ = (10, 10+txt_size[1]+10)
        img[0:HEADER_height, 0:imgW,:] = header[:,:,:]
        cv2.putText(img, msg, p1_ , cv2.FONT_HERSHEY_DUPLEX, 2, WHITE, 2)  # point is left-bottom
            
        
