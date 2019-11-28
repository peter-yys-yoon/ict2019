
from utils import *
from SPPE.src.utils.eval import getPrediction, getMultiPeakPrediction
from pPose_nms import pose_nms
from math import isnan
from matching import candidate_reselect as matching

class PersonClass:
    def __init__(self):
        self.person_next_id = 0
        self.person_trajectory_dict = {}
        self.person_list_list = []
    
    def person_tracking(self, boxes, scores, hm_data, pt1, pt2, img_id):

        person_list = []
        
        if boxes is None:
            self.person_list_list.append([])
            return person_list

        if opt.matching:  # TODO Check the difference,
            preds = getMultiPeakPrediction(
                hm_data, pt1.numpy(), pt2.numpy(), opt.inputResH, opt.inputResW, opt.outputResH,
                opt.outputResW)
            # result = matching(boxes, scores.numpy(), preds)
            result = matching(boxes, scores, preds)
        else:
            preds_hm, preds_img, preds_scores = getPrediction(hm_data, pt1, pt2, opt.inputResH,
                                                              opt.inputResW, opt.outputResH,
                                                              opt.outputResW)
            result = pose_nms(boxes, scores, preds_img, preds_scores)  # list type
            # result = {  'keypoints': ,  'kp_score': , 'proposal_score': ,  'bbox' }

        if img_id > 0:  # First frame does not have previous frame
            person_list_prev_frame = self.person_list_list[img_id - 1].copy()
        else:
            person_list_prev_frame = []
        # print(result)
        
        num_dets = len(result)
        for det_id in range(num_dets):  # IOU tracking for detections in current frame.
            # detections for current frame, obtain bbox position and track id

            result_box = result[det_id]
            kp_score = result_box['kp_score']
            if opt.matching:
                proposal_score = result_box['proposal_score']
            else:
                proposal_score = result_box['proposal_score'].numpy()[0]
                
            
            if proposal_score < 0.2:  # TODO check person proposal threshold
                continue
            
            if isnan(proposal_score):
                continue

            keypoints = result_box['keypoints']  # torch, (17,2)
            keypoints_pf = np.zeros((15, 2))

            idx_list = [16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9, 0, 0, 0]
            for i, idx in enumerate(idx_list):
                keypoints_pf[i] = keypoints[idx]
            keypoints_pf[12] = (keypoints[5] + keypoints[6]) / 2  # neck
            
            keypoints_norm = keypoints_pf - keypoints_pf[12]
            

            # COCO-order {0-nose    1-Leye    2-Reye    3-Lear    4Rear    5-Lsho    6-Rsho    7-Lelb    8-Relb    9-Lwri    10-Rwri    11-Lhip    12-Rhip    13-Lkne    14-Rkne    15-Lank    16-Rank}　
            # PoseFLow order  #{0-Rank    1-Rkne    2-Rhip    3-Lhip    4-Lkne    5-Lank    6-Rwri    7-Relb    8-Rsho    9-Lsho   10-Lelb    11-Lwri    12-neck  13-nose　14-TopHead}
            bbox_det = bbox_from_keypoints(keypoints)  # xxyy

            # enlarge bbox by 20% with same center position
            bbox_in_xywh = enlarge_bbox(bbox_det, enlarge_scale)
            bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)

            # # update current frame bbox
            if img_id == 0:  # First frame, all ids are assigned automatically
                track_id = self.person_next_id
                self.person_next_id += 1
            else:
                track_id, match_index = get_track_id_SpatialConsistency(bbox_det, person_list_prev_frame)
                if track_id != -1:  # if candidate from prev frame matched, prevent it from matching another
                    del person_list_prev_frame[match_index]

            person_det_dict = {"img_id": img_id,
                               "det_id": det_id,
                               "track_id": track_id,
                               "bbox": bbox_det,
                               "keypoints": keypoints,
                               'kp_norm': keypoints_norm,
                               'kp_poseflow': keypoints_pf,
                               'kp_score': kp_score,
                               'proposal_score': proposal_score}

            person_list.append(person_det_dict)

        num_dets = len(person_list)
        for det_id in range(num_dets):  # if IOU tracking failed, run pose matching tracking.
            person_dict = person_list[det_id]

            if person_dict["track_id"] == -1:  # this id means matching not found yet
                # track_id = bbox_det_dict["track_id"]
                track_id, match_index = get_track_id_SGCN(person_dict["bbox"], person_list_prev_frame,
                                                          person_dict["kp_poseflow"])

                if track_id != -1:  # if candidate from prev frame matched, prevent it from matching another
                    del person_list_prev_frame[match_index]
                    person_dict["track_id"] = track_id
                else:
                    # if still can not find a match from previous frame, then assign a new id
                    # if track_id == -1 and not bbox_invalid(bbox_det_dict["bbox"]):
                    person_dict["track_id"] = self.person_next_id
                    self.person_next_id += 1


        self.person_list_list.append(person_list)
        return person_list

    def person_tracjectory(self, det_list):
        for det in det_list:
            track_id = det['track_id']
            keypoint = det['kp_norm']
            if track_id in self.person_trajectory_dict.keys():
                person_dict = self.person_trajectory_dict[track_id]
                tracklet = person_dict[KEYPOINT_TRACKLET]
                tracklet.append(keypoint)
                    
                if len(tracklet) > 25: 
                    tracklet = tracklet[1:]
                    person_dict[KEYPOINT_TRACKLET] = tracklet

            else:
                person_dict = {KEYPOINT_TRACKLET: [keypoint],
                               ENERGY_HISTORY: [0],
                               ENERY_GRAPH: []
                               }

            self.person_trajectory_dict[track_id] = person_dict



    def fight_detection(self, det_list, img_id):
        for det in det_list:
            track_id = det['track_id']
            person_dict = self.person_trajectory_dict[track_id]
            tracklet = person_dict[KEYPOINT_TRACKLET]
            history = person_dict[ENERGY_HISTORY]
            energygraph = person_dict[ENERY_GRAPH]
            if len(tracklet) > 1:
                # print(track_id , tracklet)
                
                enegry = get_nonzero_std(tracklet)
                history.append(enegry)
                energygraph.append([img_id, enegry])
                
            
            if len(tracklet) > 25:
                tracklet = tracklet[1:]
            if len(history) > 30:
                history = history[1:]

            person_dict[CAR_TRACKLET] = tracklet
            person_dict[MOVE_HISTORY] = history 