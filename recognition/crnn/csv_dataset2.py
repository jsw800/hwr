import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

from collections import defaultdict
import os
import cv2
import numpy as np

import random
import string_utils

import grid_distortion


class HwDataset(Dataset):
    def __init__(self, csv_path, char_to_idx, img_height=32, root_path=".", augmentation=False):
        with open(csv_path) as f:
            data = [row.split('\t') for row in f.read().split('\n') if row != ''][1:]

        self.root_path = root_path
        self.img_height = img_height
        self.char_to_idx = char_to_idx
        self.data = data
        self.augmentation = augmentation
        self.cur_img = None
        self.cur_img_name = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # cache current image so we don't waste time reopening
        if item[0] != self.cur_img_name or self.cur_img is None:
            self.cur_img_name = item[0]
            self.cur_img = cv2.imread(os.path.join(self.root_path, item[0] + '.jpg'))

        coords = [int(coord) for coord in item[7:11]]
        img = self.cur_img[coords[2]:coords[3], coords[0]:coords[1]]
        gt = item[6]

        if img is None:
            print("Warning: image is None:", os.path.join(self.root_path, item['image_path']))
            return None

        percent = float(self.img_height) / img.shape[0]
        img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation=cv2.INTER_CUBIC)

        if self.augmentation:
            img = grid_distortion.warp_image(img)

        img = img.astype(np.float32)
        img = img / 128.0 - 1.0

        gt_label = string_utils.str2label(gt, self.char_to_idx)

        return {
            "line_img": img,
            "gt_label": gt_label,
            "gt": gt
        }
