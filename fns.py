

class vehicle:
    def __init__:
        pass
    


   def parking_detection(car_dest_list, img, img_id):

        imgW, imgH = img.shape[1], img.shape[0]
        
        parking_flag = 0
        warning_flag = 0
        
        for car in car_dest_list:
            x, y, w, h = car['bbox']
            track_id = car['track_id']
            score = car['score']
            conf = car['conf']
            
            
            # here alpha.
            # This is ganzi
            
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
                    filter = np.zeros(cropped.shape, dtype=img.dtype)
                    # print(cropped.shape, filter.shape)
                    filter[:, :, :] = YELLOW
                    warning_flag += 1
                    overlayed = cv2.addWeighted(cropped, 0.8, filter, 0.2, 0)
                    img[y:y + h, x:x + w, :] = overlayed[:, :, :]
                    
                    
                    text_filled2(img, (10, 80), 'Red zone Warning', YELLOW, 2, 2)
                    if img_id % BLINK_CAR_CAUTION_INTERVAL == 0:
                        cv2.rectangle(img, (BKINK_CAR_CAUTION_BBOX_SIZE, BKINK_CAR_CAUTION_BBOX_SIZE), (imgW - BKINK_CAR_CAUTION_BBOX_SIZE, imgH - BKINK_CAR_CAUTION_BBOX_SIZE), YELLOW, BKINK_CAR_CAUTION_BBOX_SIZE)
                        


                elif moving_parking:
                    
                    cv2.rectangle(img, (x, y), (x + w, y + h), RED, 1)
                    text_filled(img, (x, y), f'CAR {track_id} STOP', RED)

                    cropped = img[y:y + h, x:x + w, :]
                    filter = np.zeros(cropped.shape, dtype=img.dtype)
                    # print(cropped.shape, filter.shape)
                    filter[:, :, :] = RED
                    overlayed = cv2.addWeighted(cropped, 0.8, filter, 0.2, 0)
                    img[y:y + h, x:x + w, :] = overlayed[:, :, :]
                    parking_flag += 1

                    text_filled2(img, (10, 80), 'Red zone Parking!!', RED, 2, 2)
                    if img_id & BLINK_CAR_WARNING_INTERVAL == 0:
                        cv2.rectangle(img, (BKINK_CAR_WARNING_BBOX_SIZE, BKINK_CAR_WARNING_BBOX_SIZE), (imgW - BKINK_CAR_WARNING_BBOX_SIZE, imgH - BKINK_CAR_WARNING_BBOX_SIZE), RED, BKINK_CAR_WARNING_BBOX_SIZE)
                        
                        
                
                
                    

                    # if track_id == 13:
                    #     cv2.rectangle(img, (x, y), (x + w, y + h), RED, 1)
                    #     text_filled(img, (x, y), f'{track_id} Parking', RED)
                    #     cropped = img[y:y + h, x:x + w, :]
                    #     filter = np.zeros(cropped.shape, dtype=img.dtype)
                    #     # print(cropped.shape, filter.shape)
                    #     filter[:, :, 2] = 255
                    #     # print(overlay.shape)
                    #     # cv2.rectangle(overlay, (0, 0), (w, h), COLOR_RED, -1)
                    #     overlayed = cv2.addWeighted(cropped, 0.8, filter, 0.2, 0)
                    #     img[y:y + h, x:x + w, :] = overlayed[:, :, :]

                    #     cv2.rectangle(img, (0, 0), (imgW - 15, imgH - 15), RED, 10)
                    #     text_filled2(img, (10, 80), 'Red zone PARKING!!', RED, 2, 2)

                    # else:
                    #     text_filled(img, (x, y), f'{track_id} Inactive', COLOR_INACTIVE)
                    #     cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_INACTIVE, 1)
                    
        return warning_flag, parking_flag