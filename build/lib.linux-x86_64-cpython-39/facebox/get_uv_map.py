# coding: utf-8

__author__ = 'cleardusk'

import sys
import argparse
import cv2
import numpy as np
import yaml

from facebox.FaceBoxes.FaceBoxes import FaceBoxes
from facebox.TDDFA import TDDFA
from facebox.utils.uv import uv_tex
from facebox.utils.pose import calc_pose
import os.path as osp
import time


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


class UVMapGetter:

    def __init__(self,
                 onnx=False,
                 gpu=True,
                 uv_dim=64
                 ):
        fp_config = make_abs_path('configs/mb1_120x120.yml')
        cfg = yaml.load(open(fp_config), Loader=yaml.SafeLoader)
        device = 'cuda' if gpu else 'cpu'

        # Init FaceBoxes and TDDFA, recommend using onnx flag
        if onnx:
            import os
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['OMP_NUM_THREADS'] = '4'

            from facebox.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
            from facebox.TDDFA_ONNX import TDDFA_ONNX

            self.tddfa = TDDFA_ONNX(**cfg)
            self.face_boxes = FaceBoxes_ONNX()
        else:

            self.tddfa = TDDFA(gpu_mode=gpu, **cfg)
            self.face_boxes = FaceBoxes(device=device)
            self.uv_dim = uv_dim

    def __call__(self, imgs):

        uvs = []
        meta = []
        times = []
        for img in imgs:
            time_sample = {}
            if img.dtype == np.uint16:
                uint8_img = ((img / (2 ** 16 - 1.)) * 255.).astype(np.uint8)
            else:
                uint8_img = img
            start = time.time()
            boxes = self.face_boxes(uint8_img)
            end = time.time()
            time_sample['face_detection'] = end - start
            n = len(boxes)
            if n == 0:
                uvs.append(np.zeros((self.uv_dim, self.uv_dim, 3)))
                meta.append((np.nan, np.nan, np.nan))
                times.append({})
                continue

            start = time.time()
            param_lst, roi_box_lst = self.tddfa(uint8_img, boxes)
            end = time.time()
            time_sample['tddfa_inference'] = end - start

            ver_lst = self.tddfa.recon_vers(param_lst,
                                            roi_box_lst,
                                            dense_flag=True)
            P, pose = calc_pose(param_lst[0])
            yaw, pitch, roll = pose

            start = time.time()
            uv = uv_tex(img,
                        ver_lst,
                        self.tddfa.tri,
                        uv_h=self.uv_dim,
                        uv_w=self.uv_dim,
                        show_flag=False,
                        wfp=None,
                        inter='nearest',
                        yaw=yaw)
            end = time.time()
            time_sample['uv_map_extraction'] = end - start
            uvs.append(uv)
            meta.append((yaw, pitch, roll))
            times.append(time_sample)

        return uvs, meta, times
