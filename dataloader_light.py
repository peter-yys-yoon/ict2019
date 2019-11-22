import os
import sys
import time
from multiprocessing import Queue as pQueue
from threading import Thread

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

from SPPE.src.utils.eval import getPrediction, getMultiPeakPrediction
from SPPE.src.utils.img import load_image, cropBox, im_to_torch
from matching import candidate_reselect as matching
from opt import opt
from pPose_nms import pose_nms
from yolo.darknet import Darknet
from yolo.preprocess import prep_image, prep_frame
from yolo.util import dynamic_write_results

from tmp import *
from queue import Queue, LifoQueue

from fn import vis_frame_tmp as vis_frame
from fn import getTime

enlarge_scale = 0.3

YOLO_DETECTION_THRES = 0.35

from tmp import get_bbox_list


class VideoLoader:
    def __init__(self, path, batchSize=1, queueSize=50):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.path = path
        stream = cv2.VideoCapture(path)

        assert stream.isOpened(), 'Cannot capture source'
        self.fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
        self.fps = stream.get(cv2.CAP_PROP_FPS)
        self.frameSize = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.stopped = False

        self.batchSize = batchSize
        self.datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
        leftover = 0
        if (self.datalen) % batchSize:
            leftover = 1
        self.num_batches = self.datalen // batchSize + leftover

        # initialize the queue used to store frames read from
        # the video file
        if opt.sp:
            self.Q = Queue(maxsize=queueSize)
        else:
            self.Q = mp.Queue(maxsize=queueSize)

    def length(self):
        return self.datalen

    def start(self):
        # start a thread to read frames from the file video stream
        if opt.sp:
            t = Thread(target=self.update, args=())
            t.daemon = True
            t.start()
        else:
            p = mp.Process(target=self.update, args=())
            p.daemon = True
            p.start()
        return self

    def update(self):
        stream = cv2.VideoCapture(self.path)
        assert stream.isOpened(), 'Cannot capture source'

        for i in range(self.num_batches):
            img = []
            orig_img = []
            im_name = []
            im_dim_list = []
            for k in range(i * self.batchSize, min((i + 1) * self.batchSize, self.datalen)):
                """
                stacking Batch frames
                """
                inp_dim = int(opt.inp_dim)
                (grabbed, frame) = stream.read()
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.Q.put((None, None, None, None))
                    print('===========================> This video get ' + str(k) + ' frames in total.')
                    sys.stdout.flush()
                    return
                # process and add the frame to the queue
                img_k, orig_img_k, im_dim_list_k = prep_frame(frame, inp_dim)
                img.append(img_k)
                orig_img.append(orig_img_k)
                im_name.append(k)
                im_dim_list.append(im_dim_list_k)

            with torch.no_grad():
                # Human Detection
                img = torch.cat(img)
                im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

            while self.Q.full():
                time.sleep(0.5)

            self.Q.put((img, orig_img, im_name, im_dim_list))

    def videoinfo(self):
        # indicate the video info
        return (self.fourcc, self.fps, self.frameSize)

    def getitem(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        return self.Q.qsize()


class DetectionLoader:
    def __init__(self, dataloder, path, batchSize=1, queueSize=1024):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not

        self.txtpath = os.path.dirname(path) + '/' + os.path.basename(path).split('.')[0] + '.txt'
        self.stopped = False
        self.dataloder = dataloder
        self.batchSize = batchSize
        self.datalen = self.dataloder.length()
        leftover = 0

        if (self.datalen) % batchSize:
            leftover = 1
        self.num_batches = self.datalen // batchSize + leftover
        # initialize the queue used to store frames read from
        # the video file
        if opt.sp:
            self.Q = Queue(maxsize=queueSize)
        else:
            self.Q = mp.Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        if opt.sp:
            t = Thread(target=self.update, args=())
            t.daemon = True
            t.start()
        else:
            p = mp.Process(target=self.update, args=())
            p.daemon = True
            p.start()
        return self

    def update(self):
        # keep looping the whole dataset
        """

        :return:
        """

        car_list_list, hm_list_list = get_bbox_list(self.txtpath)
        for i in range(self.num_batches):  # repeat

            img, orig_img, im_name, im_dim_list = self.dataloder.getitem()

            # img = (batch, frames)
            if img is None:
                self.Q.put((None, None, None, None, None, None, None))
                return
            start_time = getTime()
            # with torch.no_grad():
            # Human Detection
            # img = img.cuda()  # image ( B, 3, 608,608 )

            for k in range(len(orig_img)):  # for k-th image detection.

                im_name_k = im_name[k]
                car_list = car_list_list[im_name_k]
                hm_list = hm_list_list[im_name_k]

                if len(car_list) == 0:  # empty car
                    car_list_np = None
                else:
                    car_list_np = np.array(car_list)

                if len(hm_list):  # human not empty

                    # bbox [idx, cls, x, y, w, h, c]

                    hm_list_np = np.array(hm_list)
                    hm_boxes_k = hm_list_np[:, 0:4]

                    hm_scores_k = hm_list_np[:, 4]

                    size = hm_boxes_k.shape[0]
                    inps = torch.zeros(size, 3, opt.inputResH, opt.inputResW)
                    pt1 = torch.zeros(size, 2)
                    pt2 = torch.zeros(size, 2)
                    item = (orig_img[k], im_name[k], hm_boxes_k, hm_scores_k, inps, pt1, pt2, car_list_np)
                else:
                    item = (orig_img[k], im_name[k], None, None, None, None, None, car_list_np)  # 8-elemetns

                if self.Q.full():
                    time.sleep(0.3)
                self.Q.put(item)

                # print('--------------- car person', carperson.size())
                # print('--------------- hm dets', hm_dets.size())
                # print('--------------- class ind', class__person_mask_ind.size())
                # print()
                # car_cand = car_dets[car_dets[:, 0] == k]

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue len
        return self.Q.qsize()


class DetectionProcessor:
    def __init__(self, detectionLoader, queueSize=1024):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.detectionLoader = detectionLoader
        self.stopped = False
        self.datalen = self.detectionLoader.datalen

        # initialize the queue used to store data
        if opt.sp:
            self.Q = Queue(maxsize=queueSize)
        else:
            self.Q = pQueue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        if opt.sp:
            t = Thread(target=self.update, args=())
            t.daemon = True
            t.start()
        else:
            p = mp.Process(target=self.update, args=())
            p.daemon = True
            p.start()
        return self

    def update(self):
        # keep looping the whole dataset
        for i in range(self.datalen):

            with torch.no_grad():
                (orig_img, im_name, boxes, scores, inps, pt1, pt2, CAR) = self.detectionLoader.read()
                # print('detection processor' , im_name, boxes)

                if orig_img is None:
                    self.Q.put((None, None, None, None, None, None, None, None))
                    return

                # if boxes is None or boxes.nelement() == 0:
                if boxes is None:

                    while self.Q.full():
                        time.sleep(0.2)
                    self.Q.put((None, orig_img, im_name, boxes, scores, None, None, CAR))
                    continue

                start_time = getTime()
                inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
                inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)

                # print(boxes, pt1,pt2)
                ckpt_time, torch_time = getTime(start_time)
                # print('torch time', round(torch_time, 3))

                while self.Q.full():
                    time.sleep(0.2)
                self.Q.put((inps, orig_img, im_name, boxes, scores, pt1, pt2, CAR))

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue len
        return self.Q.qsize()


class DataWriter:
    def __init__(self, save_video=False,
                 savepath='examples/res/1.avi', fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=25, frameSize=(640, 480),
                 queueSize=1024):
        if save_video:
            # initialize the file video stream along with the boolean
            # used to indicate if the thread should be stopped or not
            # self.stream = cv2.VideoWriter(savepath, fourcc, fps, frameSize)
            self.stream = cv2.VideoWriter(savepath, fourcc, 20, frameSize)
            assert self.stream.isOpened(), 'Cannot open video for writing'
        self.save_video = save_video
        self.stopped = False
        self.final_result = []
        self.track_dict = {}
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)
        if opt.save_img:
            if not os.path.exists(opt.outputpath + '/vis'):
                os.mkdir(opt.outputpath + '/vis')

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def tracking(self, det_list):

        for det in det_list:
            track_id = det['track_id']
            xyxy = xywh_to_x1y1x2y2(det['bbox'])

            if track_id in self.track_dict.keys():
                track_obj = self.track_dict[track_id]
                tracklet = track_obj[CAR_TRACKLET]
                history = track_obj[MOVE_HISTORY]
                moved = 0
                if len(tracklet) > 1:
                    dist, diag = avg_dist(tracklet)
                    th = diag * 0.02
                    # TODO more condition,
                    # center of bbox is actually moved? |x1-x2| > 20
                    if dist > th:
                        moved = 1
                    else:
                        moved = 0
                history.append(moved)
                tracklet.append(xyxy)

                if len(tracklet) > 25:
                    tracklet = tracklet[1:]
                if len(history) > 120:
                    history = history[1:]
                # if len()

                track_obj[CAR_TRACKLET] = tracklet
                track_obj[MOVE_HISTORY] = history

            else:
                track_obj = {CAR_TRACKLET: [xyxy],
                             MOVE_HISTORY: [0],
                             GTA_HISTORY: [0]
                             }

            self.track_dict[track_id] = track_obj

    def update(self):
        next_id = 0
        car_next_id = 0
        bbox_dets_list_list = []
        keypoints_list_list = []
        car_dets_list_list = []

        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                if self.save_video:
                    self.stream.release()
                return
            # otherwise, ensure the queue is not empty

            if not self.Q.empty():
                start_time = getTime()

                (boxes, scores, hm_data, pt1, pt2, orig_img, img_id, CAR) = self.Q.get()

                orig_img = np.array(orig_img, dtype=np.uint8)
                if boxes is not None:
                    boxes = boxes.astype(np.int32)

                img = orig_img

                # text_filled2(img,(5,200),str(img_id),LIGHT_GREEN,2,2)

                bbox_dets_list = []  # keyframe: start from empty
                keypoints_list = []  # keyframe: start from empty
                # print(boxes)
                if boxes is None:  # No person detection
                    pass
                    # bbox_det_dict = {"img_id": img_id,
                    #                  "det_id": 0,
                    #                  "track_id": None,
                    #                  "bbox": [0, 0, 2, 2]}
                    # bbox_dets_list.append(bbox_det_dict)
                    #
                    # keypoints_dict = {"img_id": img_id,
                    #                   "det_id": 0,
                    #                   "track_id": None,
                    #                   "keypoints": []}
                    # keypoints_list.append(keypoints_dict)


                else:
                    if opt.matching:
                        preds = getMultiPeakPrediction(
                            hm_data, pt1.numpy(), pt2.numpy(), opt.inputResH, opt.inputResW, opt.outputResH,
                            opt.outputResW)
                        result = matching(boxes, scores.numpy(), preds)
                    else:

                        preds_hm, preds_img, preds_scores = getPrediction(hm_data, pt1, pt2, opt.inputResH,
                                                                          opt.inputResW, opt.outputResH,
                                                                          opt.outputResW)

                        # print('number of result', preds_hm,  preds_scores )
                        result = pose_nms(boxes, scores, preds_img, preds_scores)  # list type
                        # result = {  'keypoints': ,  'kp_score': , 'proposal_score': ,  'bbox' }

                    if img_id > 0:  # First frame does not have previous frame
                        bbox_list_prev_frame = bbox_dets_list_list[img_id - 1].copy()
                        keypoints_list_prev_frame = keypoints_list_list[img_id - 1].copy()
                    else:
                        bbox_list_prev_frame = []
                        keypoints_list_prev_frame = []

                    # boxes.size(0)
                    num_dets = len(result)

                    for bbox in boxes:
                        x, y, w, h = bbox.astype(np.uint32)
                        cv2.rectangle(orig_img, (x, y), (x + w, y + h), (253, 222, 111), 1)

                    for det_id in range(num_dets):  # IOU tracking for detections in current frame.
                        # detections for current frame
                        # obtain bbox position and track id

                        result_box = result[det_id]
                        kp_score = result_box['kp_score']
                        proposal_score = result_box['proposal_score'].numpy()[0]
                        if proposal_score < 1.3:
                            continue

                        keypoints = result_box['keypoints']  # torch, (17,2)
                        keypoints_pf = np.zeros((15, 2))

                        idx_list = [16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9, 0, 0, 0]
                        for i, idx in enumerate(idx_list):
                            keypoints_pf[i] = keypoints[idx]
                        keypoints_pf[12] = (keypoints[5] + keypoints[6]) / 2  # neck

                        # COCO-order {0-nose    1-Leye    2-Reye    3-Lear    4Rear    5-Lsho    6-Rsho    7-Lelb    8-Relb    9-Lwri    10-Rwri    11-Lhip    12-Rhip    13-Lkne    14-Rkne    15-Lank    16-Rank}　
                        # PoseFLow order  #{0-Rank    1-Rkne    2-Rhip    3-Lhip    4-Lkne    5-Lank    6-Rwri    7-Relb    8-Rsho    9-Lsho   10-Lelb    11-Lwri    12-neck  13-nose　14-TopHead}

                        bbox_det = bbox_from_keypoints(keypoints)  # xxyy

                        # bbox_in_xywh = enlarge_bbox(bbox_det, enlarge_scale)
                        # bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)

                        # Keyframe: use provided bbox
                        # if bbox_invalid(bbox_det):
                        #     track_id = None  # this id means null
                        #     keypoints = []
                        #     bbox_det = [0, 0, 2, 2]
                        #     # update current frame bbox
                        #     bbox_det_dict = {"img_id": img_id,
                        #                      "det_id": det_id,
                        #                      "track_id": track_id,
                        #                      "bbox": bbox_det}
                        #     bbox_dets_list.append(bbox_det_dict)
                        #     # update current frame keypoints
                        #     keypoints_dict = {"img_id": img_id,
                        #                       "det_id": det_id,
                        #                       "track_id": track_id,
                        #                       "keypoints": keypoints}
                        #     keypoints_list.append(keypoints_dict)
                        #     continue

                        # # update current frame bbox

                        if img_id == 0:  # First frame, all ids are assigned automatically
                            track_id = next_id
                            next_id += 1
                        else:
                            track_id, match_index = get_track_id_SpatialConsistency(bbox_det, bbox_list_prev_frame)
                            # print('track' ,track_id, match_index)

                            if track_id != -1:  # if candidate from prev frame matched, prevent it from matching another
                                del bbox_list_prev_frame[match_index]
                                del keypoints_list_prev_frame[match_index]

                        # update current frame bbox
                        bbox_det_dict = {"img_id": img_id,
                                         "det_id": det_id,
                                         "track_id": track_id,
                                         "bbox": bbox_det}

                        # update current frame keypoints
                        keypoints_dict = {"img_id": img_id,
                                          "det_id": det_id,
                                          "track_id": track_id,
                                          "keypoints": keypoints,
                                          'kp_poseflow': keypoints_pf,
                                          'kp_score': kp_score,
                                          'bbox': bbox_det,
                                          'proposal_score': proposal_score}

                        bbox_dets_list.append(bbox_det_dict)
                        keypoints_list.append(keypoints_dict)

                    num_dets = len(bbox_dets_list)
                    for det_id in range(num_dets):  # if IOU tracking failed, run pose matching tracking.
                        bbox_det_dict = bbox_dets_list[det_id]
                        keypoints_dict = keypoints_list[det_id]

                        # assert (det_id == bbox_det_dict["det_id"])
                        # assert (det_id == keypoints_dict["det_id"])

                        if bbox_det_dict["track_id"] == -1:  # this id means matching not found yet
                            # track_id = bbox_det_dict["track_id"]
                            track_id, match_index = get_track_id_SGCN(bbox_det_dict["bbox"], bbox_list_prev_frame,
                                                                      keypoints_dict["kp_poseflow"],
                                                                      keypoints_list_prev_frame)

                            if track_id != -1:  # if candidate from prev frame matched, prevent it from matching another
                                del bbox_list_prev_frame[match_index]
                                del keypoints_list_prev_frame[match_index]
                                bbox_det_dict["track_id"] = track_id
                                keypoints_dict["track_id"] = track_id

                            # if still can not find a match from previous frame, then assign a new id
                            # if track_id == -1 and not bbox_invalid(bbox_det_dict["bbox"]):
                            if track_id == -1:
                                bbox_det_dict["track_id"] = next_id
                                keypoints_dict["track_id"] = next_id
                                next_id += 1

                    # update frame
                    # print('keypoint list', len(keypoints_list))
                    vis_frame(img, keypoints_list)

                """
                Car
                """

                if CAR is not None:
                    car_np = CAR
                    new_car_bboxs = car_np[:, 0:4].astype(np.uint32)  # b/  x y w h c / cls_conf, cls_idx
                    new_car_score = car_np[:, 4]
                    cls_conf = car_np[:, 4]

                    # print("id: ", img_id , " ------------ " , new_car_bboxs, new_car_score)
                    # cls_conf = car_np[:, 6]
                    car_dest_list = []

                    if img_id > 1:  # First frame does not have previous frame
                        car_bbox_list_prev_frame = car_dets_list_list[img_id - 1].copy()
                    else:
                        car_bbox_list_prev_frame = []

                    # print('car bbox list prev frame ', len(car_bbox_list_prev_frame))
                    for c, score, conf in zip(new_car_bboxs, new_car_score, cls_conf):
                        # car_bbox_det = c
                        # car_bbox_det = x1y1x2y2_to_xywh(c)
                        bbox_det = c
                        # bbox_in_xywh = enlarge_bbox(car_bbox_det, enlarge_scale)
                        # bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)

                        if img_id == 0:  # First frame, all ids are assigned automatically
                            car_track_id = car_next_id
                            car_next_id += 1
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
                            car_bbox_det_dict["track_id"] = car_next_id
                            car_next_id += 1

                    self.tracking(car_dest_list)
                    car_dets_list_list.append(car_dest_list)

                else:
                    car_dest_list = []
                    bbox_det_dict = {"img_id": img_id,
                                     "det_id": 0,
                                     "track_id": None,
                                     "bbox": [0, 0, 2, 2],
                                     "score": 0,
                                     "conf": 0}
                    car_dest_list.append(bbox_det_dict)
                    car_dets_list_list.append(car_dest_list)

                bbox_dets_list_list.append(bbox_dets_list)
                keypoints_list_list.append(keypoints_list)

                if img_id != 0:
                    self.car_person_detection(car_dest_list, bbox_dets_list, img)
                    self.car_parking_detection(car_dest_list, img, img_id)

                ckpt_time, det_time = getTime(start_time)
                cv2.putText(img, str(1 / det_time), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
                if opt.vis:
                    cv2.imshow("AlphaPose Demo", img)
                    cv2.waitKey(33)
                if opt.save_video:
                    self.stream.write(img)
            else:
                time.sleep(0.1)

    def car_person_detection(self, car_dets_list, hm_dets_list, img):

        for car in car_dets_list:
            car_track_id = car['track_id']
            if car_track_id is None:
                continue

            car_bbox = car['bbox']
            for human in hm_dets_list:
                human_track_id = human['track_id']
                if human_track_id is None:
                    continue
                hum_bbox = human['bbox']
                boxa = xywh_to_x1y1x2y2(hum_bbox)
                boxb = xywh_to_x1y1x2y2(car_bbox)
                x, y, w, h = x1y1x2y2_to_xywh(boxa)
                area = iou(boxa, boxb)

                if area > 0.02:
                    cropped = img[y:y + h, x:x + w, :]
                    filter = np.zeros(cropped.shape, dtype=img.dtype)
                    filter[:, :, 2] = 255
                    overlayed = cv2.addWeighted(cropped, 0.9, filter, 0.1, 0)
                    img[y:y + h, x:x + w, :] = overlayed[:, :, :]

    def car_parking_detection(self, car_dest_list, img, img_id):

        imgW, imgH = img.shape[1], img.shape[1]
        for car in car_dest_list:
            x, y, w, h = car['bbox']
            track_id = car['track_id']
            score = car['score']
            conf = car['conf']
            LAST_MOVED_COUNT = -180
            tracker = self.track_dict[track_id]
            history = tracker[MOVE_HISTORY]
            moved = np.sum(history[-10:])
            last_moved = np.sum(history[LAST_MOVED_COUNT:])

            COLOR_MOVING = (0, 255, 0)
            COLOR_RED = (0, 0, 255)

            COLOR_INACTIVE = (255, 0, 0)
            YELLOW = (15, 217, 255)
            ORANGE = (0, 129, 255)
            # print('car parking detction', x, y, w, h)
            cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_INACTIVE, 1)
            text_filled(img, (x, y), f'{track_id} Inactive', COLOR_INACTIVE)

            #############################################
            ################ TODO GTA
            if track_id == 4:
                if 295 < img_id < 420:
                    cv2.rectangle(img, (x, y), (x + w, y + h), ORANGE, 1)
                    text_filled(img, (x, y), f'CAR {track_id} Inactive', ORANGE)

                    cropped = img[y:y + h, x:x + w, :]
                    filter = np.zeros(cropped.shape, dtype=img.dtype)
                    # print(cropped.shape, filter.shape)
                    filter[:, :, :] = ORANGE
                    # cv2.rectangle(overlay, (0, 0), (w, h), COLOR_RED, -1)
                    overlayed = cv2.addWeighted(cropped, 0.9, filter, 0.1, 0)
                    img[y:y + h, x:x + w, :] = overlayed[:, :, :]

                    cv2.rectangle(img, (3, 3), (imgW - 10, imgH - 20), ORANGE, 10)
                    text_filled2(img, (10, 80), 'Suspicious!!', ORANGE, 2, 2)

                if img_id > 419:
                    cv2.rectangle(img, (x, y), (x + w, y + h), RED, 1)
                    text_filled(img, (x, y), f'{track_id} Inactive', RED)
                    cropped = img[y:y + h, x:x + w, :]
                    filter = np.zeros(cropped.shape, dtype=img.dtype)
                    # print(cropped.shape, filter.shape)
                    filter[:, :, 2] = 255
                    # print(overlay.shape)
                    # cv2.rectangle(overlay, (0, 0), (w, h), COLOR_RED, -1)
                    overlayed = cv2.addWeighted(cropped, 0.9, filter, 0.1, 0)
                    img[y:y + h, x:x + w, :] = overlayed[:, :, :]

                    cv2.rectangle(img, (0, 0), (imgW - 15, imgH - 15), RED, 10)
                    text_filled2(img, (10, 80), 'Warning!!!!!!', RED, 2, 2)

    ##############################################################3
    ##############TODO Parking
    # if moved:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_MOVING, 1)
    #     text_filled(img, (x, y), f'CAR {track_id} Active', COLOR_MOVING)
    # else:
    #
    #     if last_moved:
    #         cv2.rectangle(img, (x, y), (x + w, y + h), YELLOW, 1)
    #         text_filled(img, (x, y), f'CAR {track_id} STOP', YELLOW)
    #
    #         cropped = img[y:y + h, x:x + w, :]
    #         filter = np.zeros(cropped.shape, dtype=img.dtype)
    #         # print(cropped.shape, filter.shape)
    #         filter[:, :, :] = YELLOW
    #         # cv2.rectangle(overlay, (0, 0), (w, h), COLOR_RED, -1)
    #         overlayed = cv2.addWeighted(cropped, 0.8, filter, 0.2, 0)
    #         img[y:y + h, x:x + w, :] = overlayed[:, :, :]
    #
    #         cv2.rectangle(img, (3, 3), (imgW - 10, imgH - 20), YELLOW, 10)
    #         text_filled2(img, (10, 80), 'Red zone Stop!!', YELLOW, 2, 2)
    #
    #
    #     else:
    #
    #         if track_id == 13:
    #             cv2.rectangle(img, (x, y), (x + w, y + h), RED, 1)
    #             text_filled(img, (x, y), f'{track_id} Parking', RED)
    #             cropped = img[y:y + h, x:x + w, :]
    #             filter = np.zeros(cropped.shape, dtype=img.dtype)
    #             # print(cropped.shape, filter.shape)
    #             filter[:, :, 2] = 255
    #             # print(overlay.shape)
    #             # cv2.rectangle(overlay, (0, 0), (w, h), COLOR_RED, -1)
    #             overlayed = cv2.addWeighted(cropped, 0.8, filter, 0.2, 0)
    #             img[y:y + h, x:x + w, :] = overlayed[:, :, :]
    #
    #             cv2.rectangle(img, (0, 0), (imgW - 15, imgH - 15), RED, 10)
    #             text_filled2(img, (10, 80), 'Red zone PARKING!!', RED, 2, 2)
    #
    #         else:
    #             text_filled(img, (x, y), f'{track_id} Inactive', COLOR_INACTIVE)
    #             cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_INACTIVE, 1)

    def running(self):
        # indicate that the thread is still running
        time.sleep(0.2)
        return not self.Q.empty()

    def save(self, boxes, scores, hm_data, pt1, pt2, orig_img, im_name, CAR):
        # save next frame in the queue
        self.Q.put((boxes, scores, hm_data, pt1, pt2, orig_img, im_name, CAR))

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        time.sleep(0.2)

    def results(self):
        # return final result
        return self.final_result

    def len(self):
        # return queue len
        return self.Q.qsize()


def crop_from_dets(img, boxes, inps, pt1, pt2):
    '''
    Crop human from origin image according to Dectecion Results
    '''

    imght = img.size(1)
    imgwidth = img.size(2)
    tmp_img = img
    tmp_img[0].add_(-0.406)
    tmp_img[1].add_(-0.457)
    tmp_img[2].add_(-0.480)
    for i, box in enumerate(boxes):
        upLeft = torch.Tensor((float(box[0]), float(box[1])))
        bottomRight = torch.Tensor((float(box[2]), float(box[3])))

        ht = bottomRight[1] - upLeft[1]
        width = bottomRight[0] - upLeft[0]

        scaleRate = 0.3

        upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
        upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
        bottomRight[0] = max(min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
        bottomRight[1] = max(min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

        try:
            inps[i] = cropBox(tmp_img.clone(), upLeft, bottomRight, opt.inputResH, opt.inputResW)
            # print(upLeft,bottomRight, inps[i].size())
        except IndexError:
            print(tmp_img.shape)
            print(upLeft)
            print(bottomRight)
            print('===')
        pt1[i] = upLeft
        pt2[i] = bottomRight

    return inps, pt1, pt2
