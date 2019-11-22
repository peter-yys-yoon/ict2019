'''
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    November 5th, 2018

    Load keypoints from existing openSVAI data format
    and turn these keypoints into Graph structure for GCN

    Perform pose matching on these pairs.
    Output the image indicating whther they match or not.
'''
# import numpy as np
# import argparse
# import torch
#
# import sys, os
# import cv2
# sys.path.append(os.path.abspath("../"))
# sys.path.append(os.path.abspath("utils"))
# sys.path.append(os.path.abspath("visualizer"))
# sys.path.append(os.path.abspath("graph"))
#
# # from utils_json import *
# # from utils_io_folder import *
#
# # from keypoint_visualizer import *
# # from detection_visualizer import *
#
# #----------------------------------------------------
# from gcn_utils.io import IO
# from gcn_utils.gcn_model import Model
# from gcn_utils.processor_siamese_gcn import SGCN_Processor
# import torchlight
#
# #class Pose_Matcher(IO):
# class Pose_Matcher(SGCN_Processor):
#     def __init__(self, argv=None):
#         self.load_arg(argv)
#         self.init_environment()
#         self.load_model()
#         self.load_weights()
#         self.gpu()
#         return
#
#
#     @staticmethod
#     def get_parser(add_help=False):
#         parent_parser = IO.get_parser(add_help=False)
#         parser = argparse.ArgumentParser(
#             add_help=False,
#             parents=[parent_parser],
#             description='Graph Convolution Network for Pose Matching')
#         #parser.set_defaults(config='config/inference.yaml')
#         parser.set_defaults(config='graph/config/inference.yaml')
#         return parser
#
#
#     def inference(self, data_1, data_2):
#         self.model.eval()
#
#         with torch.no_grad():
#             data_1 = torch.from_numpy(data_1)
#             data_1 = data_1.unsqueeze(0)
#             data_1 = data_1.float().to(self.dev)
#
#             data_2 = torch.from_numpy(data_2)
#             data_2 = data_2.unsqueeze(0)
#             data_2 = data_2.float().to(self.dev)
#
#             feature_1, feature_2 = self.model.forward(data_1, data_2)
#
#         # euclidian distance
#         diff = feature_1 - feature_2
#         dist_sq = torch.sum(pow(diff, 2), 1)
#         dist = torch.sqrt(dist_sq)
#
#         margin = 0.2
#         distance = dist.data.cpu().numpy()[0]
#         print("_____ Pose Matching: [dist: {:04.2f}]". format(distance))
#         if dist >= margin:
#             return False, distance  # Do not match
#         else:
#             return True, distance # Match
