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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # get filename from correct subdirectory
        img = cv2.imread("filename")

        gt = item[1]

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
