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


from SPPE.src.utils.img import load_image, cropBox, im_to_torch
from matching import candidate_reselect as matching
from opt import opt

from yolo.darknet import Darknet
from yolo.preprocess import prep_image, prep_frame
from yolo.util import dynamic_write_results

from utils import *
from queue import Queue, LifoQueue

from fn import vis_frame_tmp as vis_frame
from fn import getTime

from Vehicle import VehicleClass
from Person import PersonClass




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
                time.sleep(1)

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
            start_time = getTime()
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
                            time.sleep(0.5)
                        self.Q.put((orig_img[k], im_name[k], None, None, None, None, None, None))  # 8 elements
                    continue

                ckpt_time, det_time = getTime(start_time)

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

                ckpt_time, masking_time = getTime(ckpt_time)

            hm_boxes, hm_scores = None, None

            if hm_dets.size(0) > 0:
                hm_boxes = hm_dets[:, 1:5]
                hm_scores = hm_dets[:, 5:6]

            car_box_conf = None
            if car_dets.size(0) > 0:
                car_box_conf = car_dets

            for k in range(len(orig_img)):  # for k-th image detection.

                if car_box_conf is None:
                    car_k = None
                else:
                    car_k = car_box_conf[car_box_conf[:, 0] == k].numpy()
                    car_k = car_k[np.where(car_k[:, 5] > 0.2)]  # TODO check here, cls or bg/fg confidence?
                    # car_k = non_max_suppression_fast(car_k, overlapThresh=0.3)  # TODO check here, NMS

                if hm_boxes is not None:
                    hm_boxes_k = hm_boxes[hm_dets[:, 0] == k]
                    hm_scores_k = hm_scores[hm_dets[:, 0] == k]
                    inps = torch.zeros(hm_boxes_k.size(0), 3, opt.inputResH, opt.inputResW)
                    pt1 = torch.zeros(hm_boxes_k.size(0), 2)
                    pt2 = torch.zeros(hm_boxes_k.size(0), 2)
                    item = (orig_img[k], im_name[k], hm_boxes_k, hm_scores_k, inps, pt1, pt2, car_k)
                    # print('video processor ', 'image' , im_name[k] , 'hm box ' , hm_boxes_k.size())
                else:
                    item = (orig_img[k], im_name[k], None, None, None, None, None, car_k)  # 8-elemetns

                if self.Q.full():
                    time.sleep(0.5)
                self.Q.put(item)

            ckpt_time, distribute_time = getTime(ckpt_time)

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
        
        self.vehicle = VehicleClass()
        self.person = PersonClass()
        
        # initialize the queue used to store frames read from video file
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


    def update(self):
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
                (boxes, scores, hm_data, pt1, pt2, orig_img, img_id, car_np) = self.Q.get()
                orig_img = np.array(orig_img, dtype=np.uint8)
                img = orig_img

                """ PERSON """
                person_list = self.person.person_tracking(boxes, scores, hm_data, pt1, pt2, img_id)
                self.person.person_tracjectory(person_list)
                vis_frame(img, person_list)

                """ Car """
                car_dest_list = self.vehicle.car_tracking(car_np, img_id)
                self.vehicle.car_trajectory(car_dest_list)
                
                if opt.gta:
                    self.vehicle.parking_detection(car_dest_list, img, img_id)
                    
                elif opt.fight:
                    self.person.fight_detection(person_list, img_id)
                    cv2.putText(img, f'Frame: {str(img_id).ljust(4)}', (10,40) , cv2.FONT_HERSHEY_DUPLEX, 2, WHITE, 2)  # point is left-bottom
                    
                # self.fight_detection(person_list)                    
                # self.car_person_detection(car_dest_list, bbox_dets_list, img)
                    

                

                ckpt_time, det_time = getTime(start_time)
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


    def write_enery(self):
        person_dict = self.person.person_trajectory_dict
        
        hist = {}
        for id in person_dict.keys():
            hist[id] = person_dict[id][ENERY_GRAPH]
        np.save(os.path.join(opt.outputpath,'history.npy'), hist)
            

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
        except IndexError:
            print(tmp_img.shape)
            print(upLeft)
            print(bottomRight)
            print('===')
        pt1[i] = upLeft
        pt2[i] = bottomRight

    return inps, pt1, pt2
