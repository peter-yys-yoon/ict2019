import os
import sys
import time
from multiprocessing import Queue as pQueue
from threading import Thread

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.utils.data as data
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

enlarge_scale = 0.3

YOLO_DETECTION_THRES = 0.35


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
                time.sleep(2)

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
    def __init__(self, dataloder, batchSize=1, queueSize=1024):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.det_model = Darknet("yolo/cfg/yolov3-spp.cfg")
        self.det_model.load_weights('models/yolo/yolov3-spp.weights')
        self.det_model.net_info['height'] = opt.inp_dim
        self.det_inp_dim = int(self.det_model.net_info['height'])
        assert self.det_inp_dim % 32 == 0
        assert self.det_inp_dim > 32
        self.det_model.cuda()
        self.det_model.eval()

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
        for i in range(self.num_batches):  # repeat
            img, orig_img, im_name, im_dim_list = self.dataloder.getitem()
            # img = (batch, frames)
            if img is None:
                self.Q.put((None, None, None, None, None, None, None))
                return

            with torch.no_grad():
                # Human Detection
                img = img.cuda()  # image ( B, 3, 608,608 )
                prediction = self.det_model(img, CUDA=True)
                # ( B, 22743, 85 ) = ( batchsize, proposal boxes, xywh+cls)
                # predictions for each B image.

                # NMS process
                carperson = dynamic_write_results(prediction, opt.confidence, opt.num_classes, nms=True,
                                                  nms_conf=opt.nms_thesh)
                if isinstance(carperson, int) or carperson.shape[0] == 0:
                    for k in range(len(orig_img)):
                        if self.Q.full():
                            time.sleep(2)
                        self.Q.put((orig_img[k], im_name[k], None, None, None, None, None, None))  # 8 elements
                    continue

                carperson = carperson.cpu()  # (1) k-th image , (7) x,y,w,h,c, cls_score, cls_index
                im_dim_list = torch.index_select(im_dim_list, 0, carperson[:, 0].long())
                scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

                # coordinate transfer
                carperson[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
                carperson[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

                carperson[:, 1:5] /= scaling_factor
                for j in range(carperson.shape[0]):
                    carperson[j, [1, 3]] = torch.clamp(carperson[j, [1, 3]], 0.0, im_dim_list[j, 0])
                    carperson[j, [2, 4]] = torch.clamp(carperson[j, [2, 4]], 0.0, im_dim_list[j, 1])

                cls_car_mask = carperson * (carperson[:, -1] == 2).float().unsqueeze(1)  # car
                class__car_mask_ind = torch.nonzero(cls_car_mask[:, -2]).squeeze()
                car_dets = carperson[class__car_mask_ind].view(-1, 8)

                cls_person_mask = carperson * (carperson[:, -1] == 0).float().unsqueeze(1)  # person
                class__person_mask_ind = torch.nonzero(cls_person_mask[:, -2]).squeeze()
                hm_dets = carperson[class__person_mask_ind].view(-1, 8)

            for k in range(len(orig_img)):  # for k-th image detection.

                car_cand = car_dets[car_dets[:, 0] == k]
                hm_cand = hm_dets[hm_dets[:, 0] == k]

                if car_cand.size(0) > 0:
                    _car_np = car_cand.numpy()
                    # car_boxes = car_cand[np.where(car_cand[:, 4] > 0.35)] # TODO check here, cls or bg/fg confidence?
                    # new_car = non_max_suppression_fast(car_boxes, overlapThresh=0.7) #TODO check here, NMS

                if hm_cand.size(0) > 0:
                    hm_boxes = hm_cand[:, 1:5]
                    hm_scores = hm_cand[:, 5:6]
                    inps = torch.zeros(hm_boxes.size(0), 3, opt.inputResH, opt.inputResW)
                    pt1 = torch.zeros(hm_boxes.size(0), 2)
                    pt2 = torch.zeros(hm_boxes.size(0), 2)
                    item = (orig_img[k], im_name[k], hm_boxes, hm_scores, inps, pt1, pt2, car_cand)
                else:
                    item = (orig_img[k], im_name[k], None, None, None, None, None, car_cand)  # 8-elemetns

                if self.Q.full():
                    time.sleep(2)
                self.Q.put(item)

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

                if orig_img is None:
                    self.Q.put((None, None, None, None, None, None, None, None))
                    return

                if boxes is None or boxes.nelement() == 0:
                    while self.Q.full():
                        time.sleep(0.2)
                    self.Q.put((None, orig_img, im_name, boxes, scores, None, None, CAR))
                    continue

                inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
                inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)

                while self.Q.full():
                    time.sleep(0.2)
                self.Q.put((inps, orig_img, im_name, boxes, scores, pt1, pt2, CAR))

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue len
        return self.Q.qsize()


class VideoDetectionLoader:
    def __init__(self, path, batchSize=4, queueSize=256):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.det_model = Darknet("yolo/cfg/yolov3-spp.cfg")
        self.det_model.load_weights('models/yolo/yolov3-spp.weights')
        self.det_model.net_info['height'] = opt.inp_dim
        self.det_inp_dim = int(self.det_model.net_info['height'])
        assert self.det_inp_dim % 32 == 0
        assert self.det_inp_dim > 32
        self.det_model.cuda()
        self.det_model.eval()

        self.stream = cv2.VideoCapture(path)
        assert self.stream.isOpened(), 'Cannot capture source'
        self.stopped = False
        self.batchSize = batchSize
        self.datalen = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        leftover = 0
        if (self.datalen) % batchSize:
            leftover = 1
        self.num_batches = self.datalen // batchSize + leftover
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)

    def length(self):
        return self.datalen

    def len(self):
        return self.Q.qsize()

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping the whole video
        for i in range(self.num_batches):
            img = []
            inp = []
            orig_img = []
            im_name = []
            im_dim_list = []
            for k in range(i * self.batchSize, min((i + 1) * self.batchSize, self.datalen)):
                (grabbed, frame) = self.stream.read()
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return
                # process and add the frame to the queue
                inp_dim = int(opt.inp_dim)
                img_k, orig_img_k, im_dim_list_k = prep_frame(frame, inp_dim)
                inp_k = im_to_torch(orig_img_k)

                img.append(img_k)
                inp.append(inp_k)
                orig_img.append(orig_img_k)
                im_dim_list.append(im_dim_list_k)

            with torch.no_grad():
                ht = inp[0].size(1)
                wd = inp[0].size(2)
                # Human Detection
                img = Variable(torch.cat(img)).cuda()
                im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
                im_dim_list = im_dim_list.cuda()

                prediction = self.det_model(img, CUDA=True)
                # NMS process
                dets = dynamic_write_results(prediction, opt.confidence,
                                             opt.num_classes, nms=True, nms_conf=opt.nms_thesh)
                if isinstance(dets, int) or dets.shape[0] == 0:
                    for k in range(len(inp)):
                        while self.Q.full():
                            time.sleep(0.2)
                        self.Q.put((inp[k], orig_img[k], None, None))
                    continue

                im_dim_list = torch.index_select(im_dim_list, 0, dets[:, 0].long())
                scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

                # coordinate transfer
                dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
                dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

                dets[:, 1:5] /= scaling_factor
                for j in range(dets.shape[0]):
                    dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                    dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
                boxes = dets[:, 1:5].cpu()
                scores = dets[:, 5:6].cpu()

            for k in range(len(inp)):
                while self.Q.full():
                    time.sleep(0.2)
                self.Q.put((inp[k], orig_img[k], boxes[dets[:, 0] == k], scores[dets[:, 0] == k]))

    def videoinfo(self):
        # indicate the video info
        fourcc = int(self.stream.get(cv2.CAP_PROP_FOURCC))
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        frameSize = (int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return (fourcc, fps, frameSize)

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


class WebcamLoader:
    def __init__(self, webcam, queueSize=256):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(int(webcam))
        assert self.stream.isOpened(), 'Cannot capture source'
        self.stopped = False
        # initialize the queue used to store frames read from
        # the video file
        self.Q = LifoQueue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return
                # process and add the frame to the queue
                inp_dim = int(opt.inp_dim)
                img, orig_img, dim = prep_frame(frame, inp_dim)
                inp = im_to_torch(orig_img)
                im_dim_list = torch.FloatTensor([dim]).repeat(1, 2)

                self.Q.put((img, orig_img, inp, im_dim_list))
            else:
                with self.Q.mutex:
                    self.Q.queue.clear()

    def videoinfo(self):
        # indicate the video info
        fourcc = int(self.stream.get(cv2.CAP_PROP_FOURCC))
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        frameSize = (int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return (fourcc, fps, frameSize)

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue size
        return self.Q.qsize()

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


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

    def tracking(self, det_list, img_id):

        for det in det_list:
            track_id = det['track_id']
            # x,y,w,h = det['bbox']
            xyxy = xywh_to_x1y1x2y2(det['bbox'])

            if track_id in self.track_dict.keys():

                track_obj = self.track_dict[track_id]
                tracklet = track_obj['tracklet']
                history = track_obj['history']

                moved = 0
                if len(tracklet) > 1:
                    dist, diag = avg_dist(tracklet)
                    th = diag * 0.02
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

                track_obj['tracklet'] = tracklet
                track_obj['history'] = history


            else:
                track_obj = {'tracklet': [xyxy],
                             'history': [0]
                             }

            self.track_dict[track_id] = track_obj

    def is_moving(self, track_id):

        moved = False

        if track_id in self.track_dict.keys():
            tracklet = self.track_dict[track_id]

            if len(tracklet) > 1:

                dist, diag = avg_dist(tracklet)
                th = diag * 0.02
                # if track_id >50:
                #     print('id', track_id, 'moved' , dist, 'th', th)

                if dist > th:
                    moved = True

        return moved

    def update(self):
        # keep looping infinitely

        frame_prev = -1
        frame_cur = 0
        img_id = -1
        next_id = 0
        bbox_dets_list_list = []
        keypoints_list_list = []
        car_dets_list_list = []

        car_next_id = 0

        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                if self.save_video:
                    self.stream.release()
                return
            # otherwise, ensure the queue is not empty

            if not self.Q.empty():

                (boxes, scores, hm_data, pt1, pt2, orig_img, img_id, CAR) = self.Q.get()
                # print(img_id)
                orig_img = np.array(orig_img, dtype=np.uint8)
                img = orig_img

                bbox_dets_list = []  # keyframe: start from empty
                keypoints_list = []  # keyframe: start from empty

                if boxes is None:  # No person detection
                    bbox_det_dict = {"img_id": img_id,
                                     "det_id": 0,
                                     "track_id": None,
                                     "bbox": [0, 0, 2, 2]}
                    bbox_dets_list.append(bbox_det_dict)

                    keypoints_dict = {"img_id": img_id,
                                      "det_id": 0,
                                      "track_id": None,
                                      "keypoints": []}
                    keypoints_list.append(keypoints_dict)

                    bbox_dets_list_list.append(bbox_dets_list)
                    keypoints_list_list.append(keypoints_list)

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
                        result = pose_nms(boxes, scores, preds_img, preds_scores)  # list type

                        # 'keypoints':
                        # 'kp_score':
                        # 'proposal_score':
                        # 'bbox'
                    #
                    # print('boexes', boxes.size(), boxes)
                    # for aa in result:
                    #     keys = aa['keypoints']
                    #     bbox2  = aa['bbox']
                    #     print('pose nms keys', keys.size())
                    #     print('pose nms, box', bbox2.size(), bbox2)
                    #
                    # _result = {
                    #     'imgname': img_id,
                    #     'result': result,
                    #     'pt1': pt1,
                    #     'pt2': pt2
                    # }

                    if img_id > 0:  # First frame does not have previous frame
                        bbox_list_prev_frame = bbox_dets_list_list[img_id - 1].copy()
                        keypoints_list_prev_frame = keypoints_list_list[img_id - 1].copy()
                    else:
                        bbox_list_prev_frame = []
                        keypoints_list_prev_frame = []

                    # boxes.size(0)
                    num_dets = len(result)
                    for det_id in range(num_dets):  # detections for current frame
                        # obtain bbox position and track id

                        result_box = result[det_id]

                        kp_score = result_box['kp_score']
                        proposal_score = result_box['proposal_score'].numpy()[0]
                        if proposal_score < 1.3:
                            continue

                        keypoints = result_box['keypoints']
                        bbox_det = bbox_from_keypoints(keypoints)  # xxyy

                        # enlarge bbox by 20% with same center position
                        # bbox_x1y1x2y2 = xywh_to_x1y1x2y2(bbox_det)
                        bbox_in_xywh = enlarge_bbox(bbox_det, enlarge_scale)
                        # print('enlared', bbox_in_xywh)
                        bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)
                        # print('converted', bbox_det)

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

                        # obtain keypoints for each bbox position in the keyframe

                        # print('img id ', img_id)

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
                        bbox_dets_list.append(bbox_det_dict)

                        # update current frame keypoints
                        keypoints_dict = {"img_id": img_id,
                                          "det_id": det_id,
                                          "track_id": track_id,
                                          "keypoints": keypoints,
                                          'kp_score': kp_score,
                                          'bbox': bbox_det,
                                          'proposal_score': proposal_score}
                        keypoints_list.append(keypoints_dict)

                    num_dets = len(bbox_dets_list)
                    for det_id in range(num_dets):  # detections for current frame
                        bbox_det_dict = bbox_dets_list[det_id]
                        keypoints_dict = keypoints_list[det_id]
                        # assert (det_id == bbox_det_dict["det_id"])
                        # assert (det_id == keypoints_dict["det_id"])

                        if bbox_det_dict["track_id"] == -1:  # this id means matching not found yet
                            track_id = bbox_det_dict["track_id"]
                            # track_id, match_index = get_track_id_SGCN(bbox_det_dict["bbox"], bbox_list_prev_frame,
                            #                                           keypoints_dict["keypoints"],
                            #                                           keypoints_list_prev_frame)

                            if track_id != -1:  # if candidate from prev frame matched, prevent it from matching another
                                del bbox_list_prev_frame[match_index]
                                del keypoints_list_prev_frame[match_index]
                                bbox_det_dict["track_id"] = track_id
                                keypoints_dict["track_id"] = track_id

                            # if still can not find a match from previous frame, then assign a new id
                            if track_id == -1 and not bbox_invalid(bbox_det_dict["bbox"]):
                                bbox_det_dict["track_id"] = next_id
                                keypoints_dict["track_id"] = next_id
                                next_id += 1

                    # update frame

                    bbox_dets_list_list.append(bbox_dets_list)
                    keypoints_list_list.append(keypoints_list)

                    # draw keypoints

                    vis_frame(img, keypoints_list)
                    # _pt1, _pt2 = _result['pt1'].numpy(), _result['pt2'].numpy()
                    # pt1 = _pt1.astype(np.uint32)
                    # pt2 = _pt2.astype(np.uint32)
                    # for p1, p2 in zip(pt1, pt2):
                    #     cv2.rectangle(img, (p1[0], p1[1]), (p2[0], p2[1]), (34, 154, 11), 1)

                if CAR is not None:  # No car detection
                    car_track_id = 0
                    car_np = CAR
                    new_car_bboxs = car_np[:, 0:4].astype(np.uint32)
                    new_car_score = car_np[:, 4]
                    car_dest_list = []

                    if img_id > 1:  # First frame does not have previous frame
                        car_bbox_list_prev_frame = car_dets_list_list[img_id - 1].copy()
                    else:
                        car_bbox_list_prev_frame = []

                    # print('car bbox list prev frame ', len(car_bbox_list_prev_frame))
                    for c, score in zip(new_car_bboxs, new_car_score):
                        car_bbox_det = c
                        bbox_in_xywh = enlarge_bbox(car_bbox_det, enlarge_scale)
                        bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)

                        # obtain keypoints for each bbox position in the keyframe

                        # print('img id ', img_id)

                        if img_id == 0:  # First frame, all ids are assigned automatically
                            car_track_id = car_next_id
                            car_next_id += 1
                            # print('if img id zero' , car_next_id)

                        else:
                            car_track_id, match_index = get_track_id_SpatialConsistency(bbox_det,
                                                                                        car_bbox_list_prev_frame)
                            # print(car_track_id, match_index)
                            if car_track_id != -1:  # if candidate from prev frame matched, prevent it from matching another
                                del car_bbox_list_prev_frame[match_index]

                        bbox_det_dict = {"img_id": img_id,
                                         "track_id": car_track_id,
                                         "bbox": bbox_det}
                        car_dest_list.append(bbox_det_dict)

                    # print()
                    num_dets = len(car_dest_list)
                    for det_id in range(num_dets):  # detections for current frame
                        car_bbox_det_dict = car_dest_list[det_id]
                        # assert (det_id == bbox_det_dict["det_id"])
                        # assert (det_id == keypoints_dict["det_id"])
                        # print(Pose_matchercar_bbox_det_dict["track_id"])
                        if car_bbox_det_dict["track_id"] == -1:  # this id means matching not found yet
                            car_bbox_det_dict["track_id"] = car_next_id
                            car_next_id += 1
                            # print('car net id ', car_next_id)

                    self.tracking(car_dest_list, img_id)

                    for car in car_dest_list:
                        x, y, w, h = car['bbox']
                        track_id = car['track_id']

                        tracker = self.track_dict[track_id]
                        history = tracker['history']
                        moved = np.sum(history[-10:])
                        last_moved = np.sum(history[-60:])

                        COLOR_MOVING = (0, 255, 0)
                        COLOR_RED = (0, 0, 255)

                        COLOR_INACTIVE = (255, 0, 0)

                        cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_INACTIVE, 1)
                        text_filled(img, (x, y), f'{track_id} Inactive', COLOR_INACTIVE)

                        # if moved:
                        #     cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_MOVING, 1)
                        #     text_filled(img, (x, y), f'CAR {track_id} Active', COLOR_MOVING)
                        # else:
                        #
                        #     if last_moved:
                        #         cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_RED, 1)
                        #         text_filled(img, (x, y), f'CAR {track_id} Standstill', COLOR_RED)
                        #
                        #         cropped = img[y:y+h, x:x+w,:]
                        #         filter = np.zeros(cropped.shape,dtype=img.dtype)
                        #         # print(cropped.shape, filter.shape)
                        #         filter[:,:,2] = 255
                        #         # print(overlay.shape)
                        #         # cv2.rectangle(overlay, (0, 0), (w, h), COLOR_RED, -1)
                        #         overlayed = cv2.addWeighted(cropped,0.8,filter,0.2,0)
                        #         img[y:y+h, x:x+w,:] = overlayed[:,:,:]
                        #     else:
                        #         cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_INACTIVE, 1)
                        #         text_filled(img, (x, y), f'{track_id} Inactive', COLOR_INACTIVE)

                    car_dets_list_list.append(car_dest_list)

                else:
                    car_dest_list = []
                    bbox_det_dict = {"img_id": img_id,
                                     "det_id": 0,
                                     "track_id": None,
                                     "bbox": [0, 0, 2, 2]}
                    car_dest_list.append(bbox_det_dict)
                    car_dets_list_list.append(car_dest_list)

                # if img_id != 0:
                #     for car in car_dets_list_list[-1]:
                #         car_track_id = car['track_id']
                #         if car_track_id is None:
                #             continue
                #
                #         car_bbox = car['bbox']
                #         for human in bbox_dets_list_list[-1]:
                #             human_track_id = human['track_id']
                #             if human_track_id is None:
                #                 continue
                #             hum_bbox = human['bbox']
                #             boxa = xywh_to_x1y1x2y2(hum_bbox)
                #             boxb = xywh_to_x1y1x2y2(car_bbox)
                #             x,y,w,h = x1y1x2y2_to_xywh(boxa)
                #             area = iou(boxa,boxb)
                #
                #             if area > 0.02:
                #                 cropped = img[y:y+h, x:x+w,:]
                #                 filter = np.zeros(cropped.shape,dtype=img.dtype)
                #                 filter[:,:,2] = 255
                #                 overlayed = cv2.addWeighted(cropped,0.9,filter,0.1,0)
                #                 img[y:y+h, x:x+w,:] = overlayed[:,:,:]

                if opt.vis:
                    cv2.imshow("AlphaPose Demo", img)
                    cv2.waitKey(1)
                if opt.save_video:
                    self.stream.write(img)
            else:
                time.sleep(0.1)

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


class Mscoco(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.img_folder = '../data/coco/images'  # root image folders
        self.is_train = train  # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.nJoints_coco = 17
        self.nJoints_mpii = 16
        self.nJoints = 33

        self.accIdxs = (1, 2, 3, 4, 5, 6, 7, 8,
                        9, 10, 11, 12, 13, 14, 15, 16, 17)
        self.flipRef = ((2, 3), (4, 5), (6, 7),
                        (8, 9), (10, 11), (12, 13),
                        (14, 15), (16, 17))

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


def int2round(src):
    """
    returns rounded integer recursively
    :param src:
    :return:
    """
    if isinstance(src, float):
        return int(round(src))

    elif isinstance(src, tuple):
        res = []
        for i in range(len(src)):
            res.append(int(round(src[i])))
        return tuple(res)

    elif isinstance(src, list):
        res = []
        for i in range(len(src)):
            res.append(int2round(src[i]))
        return res
    elif isinstance(src, int):
        return src
    if isinstance(src, str):
        return int(src)


def convert_bbox_car(img, boxes):
    imght = img.size(1)
    imgwidth = img.size(2)

    for i, box in enumerate(boxes):
        upLeft = torch.Tensor((float(box[1]), float(box[2])))
        bottomRight = torch.Tensor((float(box[3]), float(box[4])))

        ht = bottomRight[1] - upLeft[1]
        width = bottomRight[0] - upLeft[0]

        scaleRate = 0.3

        upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
        upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
        bottomRight[0] = max(min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
        bottomRight[1] = max(min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

        boxes[i, 1] = upLeft[0]
        boxes[i, 2] = upLeft[1]
        boxes[i, 3] = bottomRight[0]
        boxes[i, 4] = bottomRight[1]

    return boxes


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
