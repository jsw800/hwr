import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

from collections import defaultdict
import os
import cv2
import numpy as np
import codecs

import random
import string_utils

import grid_distortion

class HwDataset(Dataset):
    def __init__(self, csv_path, char_to_idx, img_height=32, root_path=".",
                                    augmentation=False):
        f = codecs.open(csv_path, encoding='utf-8', mode='r+')

        self.data = {}
        for i, row in enumerate(f):
            if i == 0:
                self.header = row.split('\t')
            else:
                info = row.split('\t')
                if info[0] not in self.data:
                    self.data[info[0]] = []
                self.data[info[0]].append(info)

        self.keys = self.data.keylist()
        self.root_path = root_path
        self.img_height = img_height
        self.char_to_idx = char_to_idx
        self.augmentation = augmentation

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):

        img = cv2.imread(os.path.join(self.root_path, self.data[self.keys[idx]]) + '.jpg')

        if img is None:
            print("Warning: image is None: ", os.path.join(self.root_path, item[0]))
            return None

        return {
            "img": img,
            "gts": self.data[self.keys[idx]]
        }

    def get_gts(self, org_img, row, category):

        xy = row[category+1:category+5]
        min_x, max_x, min_y, max_y = int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])
        img = org_img[min_y:max_y, min_x:max_x]

        percent = float(self.img_height) / img.shape[0]
        img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)

        if self.augmentation:
            img = grid_distortion.warp_image(img)

        img = img.astype(np.float32)
        img = img / 128.0 - 1.0

        gt = item[category]
        if gt == ' ':
            gt = '@'
        gt_label = string_utils.str2label(gt, self.char_to_idx)

        return {
            "line_img": img,
            "gt_label": gt_label,
            "gt": gt
        }


#
#
#
#
#
#
#
#
