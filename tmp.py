import sys
import os

sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath("visualizer"))
sys.path.append(os.path.abspath("graph"))

import cv2
from opt import opt
import torch.utils.data as data
from gcn_utils.io import IO
from gcn_utils.gcn_model import Model
from gcn_utils.processor_siamese_gcn import SGCN_Processor
import torchlight
import numpy as np

UPPER = 'upper'
LOWER = 'lower'
MEAN = 'mean'

##### CAR
GTA_HISTORY = 'gta_history'
MOVE_HISTORY = 'mv_history'
CAR_TRACKLET = 'car_tracklet'

FPS = 30
LAST_MOVED_MOVE_COUNT = -10
LAST_MOVED_STOP_COUNT = FPS * -4
LAST_MOVED_PARKING_COUNT = FPS * -16
THRESHOLD_CAR_TRACKLET_SIZE = 25 * FPS
THRESHOLD_CAR_HISTORY_SIZE = 10 * FPS
THRESHOLD_CAR_MOVED = 33


BLINK_CAR_CAUTION_INTERVAL = 15
BKINK_CAR_CAUTION_BBOX_SIZE = 15
BLINK_CAR_WARNING_INTERVAL = 9
BKINK_CAR_WARNING_BBOX_SIZE = 8


##### PERSON
KEYPOINT_TRACKLET='keypoint_tracklet'
ENERGY_HISTORY ='energy_history'

THRESHOLD_FIGHT =34



COLOR_MOVING = (0, 255, 0)
COLOR_RED = (0, 0, 255)

COLOR_INACTIVE = (255, 0, 0)
COLOR_YELLOW = (15, 217, 255)
COLOR_ORANGE = (0, 129, 255)


fourcc_avi = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')

import argparse
import torch

import sys, os
import cv2

# from utils_json import *
# from utils_io_folder import *

# from keypoint_visualizer import *
# from detection_visualizer import *

# ----------------------------------------------------

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
ORANGE = (0, 129, 255)
YELLOW = (15, 217, 255)
MAGENTA = (255, 0, 255)
LIGHT_GREEN = (9, 249, 17)
BALCK = (0, 0, 0)
WHITE = (255, 255, 255)
import time

PLOT_COLOR = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
# PLOT_COLOR = ['go', 'bo','ro']
PLOT_MARKER = ['-',  # 0
               '.',  # 1 dot
               ',',  # 2 pixel
               '--.',  # 3 d otted-line
               'o'  # 4 circle
               ]
VIDEO_EXT_LIST = ['mp4', 'avi']
COLOR_LIST = [RED, GREEN, BLUE, ORANGE, YELLOW, MAGENTA, LIGHT_GREEN, WHITE, BALCK]

# FONT_FACE = cv2.FONT_HERSHEY_PLAIN # samll fonr
# FONT_FACE = cv2.FONT_HERSHEY_COMPLEX
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THINKESS = 1

FRAME_COUNT_LOC = (30, 50)
FRAME_DIST_LOC = (990, 50)
FRAME_STATUS_LOC = (410, 50)


def get_nonzero_std(_ss):  # tvc
    """
    np.sum(-1) # sum all elements in last axis
    index = np.sum(-1) != 0     # true/false list if elements is not zero.
    
    """
    
    s = np.array(_ss)
    
    
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    """
    index of valid frame that xyz of joints are not zero
    """
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std()  # three channels
    else:
        s = 0
    return s


def get_bbox_list(txtpath):
    txtfile = open(txtpath, 'r')
    txt = txtfile.readlines()

    N = int(txt[-1].split(' ')[0])
    car_list_list = [[] for x in range(N + 1)]
    hm_list_list = [[] for x in range(N + 1)]

    for tx in txt:
        txx = tx.split('\n')[0]
        idx, cls, x, y, w, h, c = txx.split(' ')
        # print(N, idx)
        idx, x, y, w, h, c = int(idx), max(int(x),0), max(int(y),0), int(w), int(h), float(c)

        if cls == 'car':
            # car_list_list[idx].append([x, y, x+w, y+h, c])
            car_list_list[idx].append([x, y, w, h, c])
        else:
            hm_list_list[idx].append([x, y, w, h, c])
            # hm_list_list[idx].append([x, y, x+w, y+h, c])

    return car_list_list, hm_list_list


# class Pose_Matcher(IO):
class Pose_Matcher(SGCN_Processor):
    def __init__(self, argv=None):
        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()
        return

    @staticmethod
    def get_parser(add_help=False):
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=False,
            parents=[parent_parser],
            description='Graph Convolution Network for Pose Matching')
        # parser.set_defaults(config='config/inference.yaml')
        parser.set_defaults(config='graph/config/inference.yaml')
        return parser

    def inference(self, data_1, data_2):
        self.model.eval()

        with torch.no_grad():
            data_1 = torch.from_numpy(data_1)
            data_1 = data_1.unsqueeze(0)
            data_1 = data_1.float().to(self.dev)

            data_2 = torch.from_numpy(data_2)
            data_2 = data_2.unsqueeze(0)
            data_2 = data_2.float().to(self.dev)

            feature_1, feature_2 = self.model.forward(data_1, data_2)

        # euclidian distance
        diff = feature_1 - feature_2
        dist_sq = torch.sum(pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        margin = 0.2
        distance = dist.data.cpu().numpy()[0]
        # print("_____ Pose Matching: [dist: {:04.2f}]".format(distance))
        if dist >= margin:
            return False, distance  # Do not match
        else:
            return True, distance  # Match


global pose_matcher
pose_matcher = Pose_Matcher()


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


def x1y1x2y2_to_xywh(det):
    x1, y1, x2, y2 = det
    w, h = int(x2) - int(x1), int(y2) - int(y1)
    x1, y1 = int(x1), int(y1)
    return [x1, y1, w, h]


def xywh_to_x1y1x2y2(det):
    x1, y1, w, h = det
    x2, y2 = x1 + w, y1 + h
    return [x1, y1, x2, y2]


def bbox_invalid(bbox):
    if bbox == [0, 0, 2, 2]:
        return True
    if bbox[2] <= 0 or bbox[3] <= 0 or bbox[2] > 2000 or bbox[3] > 2000:
        return True
    return False


def enlarge_bbox(bbox, scale):
    assert (scale > 0)
    min_x, min_y, max_x, max_y = bbox
    margin_x = int(0.5 * scale * (max_x - min_x))
    margin_y = int(0.5 * scale * (max_y - min_y))
    if margin_x < 0: margin_x = 2
    if margin_y < 0: margin_y = 2

    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    min_x = max(0, min_x)
    min_y = max(0, min_y)

    width = max_x - min_x
    height = max_y - min_y

    # if max_y < 0 or max_x < 0 or width <= 0 or height <= 0 or width > 2000 or height > 2000:
    #     min_x = 0
    #     max_x = 2
    #     min_y = 0
    #     max_y = 2

    # bbox_enlarged = [min_x, min_y, width, height]
    bbox_enlarged = [min_x, min_y, max_x, max_y]
    return bbox_enlarged


def keypoints_to_graph(keypoints, bbox):
    # num_elements = len(keypoints)
    # num_keypoints = num_elements / 3
    # assert (num_keypoints == 15)

    # COCO format

    x0, y0, w, h = bbox
    flag_pass_check = True

    graph = 15 * [(0, 0)]
    for id in range(15):
        # x = keypoints[2 * id] - x0
        # y = keypoints[2 * id + 1] - y0
        x = keypoints[id, 0] - x0
        y = keypoints[id, 1] - y0

        # score = keypoints[3 * id + 2]
        graph[id] = (int(x), int(y))
    return graph, flag_pass_check


def pose_matching(graph_A_data, graph_B_data):
    flag_match, dist = pose_matcher.inference(graph_A_data, graph_B_data)
    return flag_match, dist


def get_pose_matching_score(keypoints_A, keypoints_B, bbox_A, bbox_B):
    # print(keypoints_A, type(keypoints_A))
    # print(keypoints_B, type(keypoints_B))
    # if keypoints_A == [] or keypoints_B == []:
    #     print("graph not correctly generated!")
    #     return sys.maxsize

    if bbox_invalid(bbox_A) or bbox_invalid(bbox_B):
        print("graph not correctly generated, invalid bbox !")
        return sys.maxsize

    graph_A, flag_pass_check = keypoints_to_graph(keypoints_A, bbox_A)
    if flag_pass_check is False:
        print("graph not correctly generated!, graph A")
        return sys.maxsize

    graph_B, flag_pass_check = keypoints_to_graph(keypoints_B, bbox_B)
    if flag_pass_check is False:
        print("graph not correctly generated!, graph B ")
        return sys.maxsize

    # sample_graph_pair = (graph_A, graph_B)
    # data_A, data_B = graph_pair_to_data(sample_graph_pair)
    # data_A, data_B = graph_pair_to_data(sample_graph_pair)
    data_A = graph_pair_to_data2(graph_A)
    data_B = graph_pair_to_data2(graph_B)

    start = time.time()
    flag_match, dist = pose_matching(data_A, data_B)
    end = time.time()
    return dist


def graph_pair_to_data2(graph):
    graph = np.transpose(graph, [1, 0])
    graph = np.expand_dims(graph, axis=2)
    graph = np.expand_dims(graph, axis=1)
    return graph


def graph_pair_to_data(sample_graph_pair):
    data_numpy_pair = []
    for siamese_id in range(2):
        # fill data_numpy
        data_numpy = np.zeros((2, 1, 15, 1))  # ( xy, 1 , 15 , 1 )

        pose = sample_graph_pair[:][siamese_id]
        data_numpy[0, 0, :, 0] = [x[0] for x in pose]  # x-cord
        data_numpy[1, 0, :, 0] = [x[1] for x in pose]  # y-cord
        data_numpy_pair.append(data_numpy)
    return data_numpy_pair[0], data_numpy_pair[1]


def get_track_id_SGCN(bbox_cur_frame, person_list_prev_frame, keypoints_cur_frame):
    
    min_index = None
    min_matching_score = sys.maxsize
    pose_matching_threshold = 0.5
    # global pose_matching_threshold

    # if track_id is still not assigned, the person is really missing or track is really lost
    track_id = -1

    for det_index, bbox_det_dict in enumerate(person_list_prev_frame):

        # check the pose matching score
        person_dict = person_list_prev_frame[det_index]
        # keypoints_prev_frame = keypoints_dict["keypoints"]
        keypoints_prev_frame = person_dict["kp_poseflow"]
        bbox_prev_frame = person_dict['bbox']
        if isinstance(keypoints_prev_frame, list):
            continue
        if isinstance(keypoints_cur_frame, list):
            continue
        pose_matching_score = get_pose_matching_score(keypoints_cur_frame, keypoints_prev_frame, bbox_cur_frame,
                                                      bbox_prev_frame)
        # print('[pse ,atcjomg score', pose_matching_score)

        if pose_matching_score <= pose_matching_threshold and pose_matching_score <= min_matching_score:
            # match the target based on the pose matching score
            min_matching_score = pose_matching_score
            min_index = det_index

    if min_index is None:
        return -1, None
    else:
        track_id = person_list_prev_frame[min_index]["track_id"]
        return track_id, min_index


def get_track_id_SpatialConsistency(bbox_cur_frame, bbox_list_prev_frame):
    thresh = 0.1
    max_iou_score = 0
    max_index = -1

    for bbox_index, bbox_det_dict in enumerate(bbox_list_prev_frame):
        bbox_prev_frame = bbox_det_dict["bbox"]

        boxA = xywh_to_x1y1x2y2(bbox_cur_frame)
        boxB = xywh_to_x1y1x2y2(bbox_prev_frame)
        # boxA = bbox_cur_frame
        # boxB = bbox_prev_frame
        iou_score = iou(boxA, boxB)
        # print(iou_score)
        if iou_score > max_iou_score:
            max_iou_score = iou_score
            max_index = bbox_index
    # print('-----')
    if max_iou_score > thresh:
        track_id = bbox_list_prev_frame[max_index]["track_id"]
        return track_id, max_index
    else:
        return -1, None


# import the necessary packages
import numpy as np


def avg_dist(_trackelt):
    trackelt = np.array(_trackelt)  # xyxy
    # print('tracklet size', trackelt.shape)
    dist = []

    # print('trackelt[0:2] .shape', trackelt[:,0:2].shape)
    cxy = trackelt[:, 0:2] + trackelt[:, 2:4]

    num = len(cxy)

    diag = np.linalg.norm(trackelt[:, 0:2] - trackelt[:, 2:4], axis=1)
    diag = np.mean(diag)

    # for tr in trackelt:
    #     a = tr[:,0:2]
    #     b = tr[:,2:4]
    #     diag = np.linalg.norm(a-b)

    for i in range(1, num):
        prev = trackelt[i - 1]
        curr = trackelt[i]
        dist.append(np.linalg.norm(prev - curr))

        # x1,y1 = trackelt[i-1]
        # x2,y2  = trackelt[i]
        # dist.append(np.linalg.norm((x1-x2,y1-y2)))

    return np.mean(dist), diag


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list

    if len(boxes) == 0:
        # return []
        return None

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    # return boxes[pick].astype(np.uint32)
    return boxes[pick]


def iou(boxA, boxB):
    # box: (x1, y1, x2, y2)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def prepare_results(test_data, cls_skeleton, cls_dets):
    cls_partsco = cls_skeleton[:, :, 2].copy().reshape(-1, cfg.nr_skeleton)

    cls_scores = 1
    dump_results = []
    cls_skeleton = np.concatenate(
        [cls_skeleton.reshape(-1, cfg.nr_skeleton * 3), (cls_scores * cls_partsco.mean(axis=1))[:, np.newaxis]],
        axis=1)
    for i in range(len(cls_skeleton)):
        result = dict(image_id=test_data['img_id'],
                      category_id=1,
                      score=float(round(cls_skeleton[i][-1], 4)),
                      keypoints=cls_skeleton[i][:-1].round(3).tolist())
        dump_results.append(result)
    return dump_results


nms_method = 'nms'
nms_thresh = 1.
min_scores = 1e-10
min_box_size = 0.
flag_nms = False  # Default is False, unless you know what you are doing


def inference_keypoints(pose_estimator, test_data):
    cls_dets = test_data["bbox"]
    # nms on the bboxes
    if flag_nms is True:
        cls_dets, keep = apply_nms(cls_dets, nms_method, nms_thresh)
        test_data = np.asarray(test_data)[keep]
        if len(keep) == 0:
            return -1
    else:
        test_data = [test_data]

    # crop and detect pose
    pose_heatmaps, details, cls_skeleton, crops, start_id, end_id = get_pose_from_bbox(pose_estimator, test_data, cfg)
    # get keypoint positions from pose
    keypoints = get_keypoints_from_pose(pose_heatmaps, details, cls_skeleton, crops, start_id, end_id)
    # dump results
    results = prepare_results(test_data[0], keypoints, cls_dets)
    return results


def bbox_from_keypoints(keys):
    # a = [x for x in range(10)]
    # a = np.array(a).reshape((5,2))
    a = keys.numpy()
    x_max = np.max(a[:, 0])
    y_max = np.max(a[:, 1])

    x_min = np.min(a[:, 0])
    y_min = np.min(a[:, 1])

    return (x_min, y_min, x_max, y_max)


# bbox_from_keypoints(3)

def frame_indexing(frame, idx):
    cv2.putText(frame, 'idx ' + str(idx), FRAME_COUNT_LOC, FONT_FACE, FONT_SCALE, WHITE, FONT_THINKESS)  # frame


def text_filled(frame, p1, label, color):
    txt_size, baseLine1 = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, FONT_SCALE, FONT_THINKESS)
    p1_ = (p1[0], p1[1])
    p2 = (p1[0] + txt_size[0], p1[1] - txt_size[1])
    # p1_ = (p1[0] - 10, p1[1] + 10)
    # p2 = (p1[0] + txt_size[0] + 10, p1[1] - txt_size[1] - 10)
    cv2.rectangle(frame, p1_, p2, color, -1)
    cv2.putText(frame, label, p1, cv2.FONT_HERSHEY_DUPLEX, FONT_SCALE, WHITE, FONT_THINKESS)  # point is left-bottom


def text_filled2(frame, p1, label, color, font_sacle, thickness):
    txt_size, baseLine1 = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_sacle, thickness)
    p1_ = (p1[0], p1[1] + baseLine1)  # left-bottom
    p2 = (p1[0] + txt_size[0], p1[1] - txt_size[1])  # top_right
    cv2.rectangle(frame, p1_, p2, color, -1)
    cv2.putText(frame, label, p1, cv2.FONT_HERSHEY_DUPLEX, font_sacle, WHITE, thickness)  # point is left-bottom


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
