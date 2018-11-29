
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

PADDING_CONSTANT = 0

def collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) < 1:
        return None
    #These all should be the same size or error
    assert len(set([b['line_img'].shape[0] for b in batch])) == 1
    assert len(set([b['line_img'].shape[2] for b in batch])) == 1

    dim0 = batch[0]['line_img'].shape[0]
    dim1 = max([b['line_img'].shape[1] for b in batch])
    dim2 = batch[0]['line_img'].shape[2]

    all_labels = []
    label_lengths = []

    input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)
    for i in xrange(len(batch)):
        b_img = batch[i]['line_img']
        input_batch[i,:,:b_img.shape[1],:] = b_img

        l = batch[i]['gt_label']
        all_labels.append(l)
        label_lengths.append(len(l))

    all_labels = np.concatenate(all_labels)
    label_lengths = np.array(label_lengths)

    line_imgs = input_batch.transpose([0,3,1,2])
    line_imgs = torch.from_numpy(line_imgs)
    labels = torch.from_numpy(all_labels.astype(np.int32))
    label_lengths = torch.from_numpy(label_lengths.astype(np.int32))

    return {
        "line_imgs": line_imgs,
        "labels": labels,
        "label_lengths": label_lengths,
        "gt": [b['gt'] for b in batch],
        "idx": batch[0]['idx']
    }

class HwDataset(Dataset):
    def __init__(self, csv_path, char_to_idx, img_height=32, root_path=".",
                                    augmentation=False, categories=[31, 36, 41],
                                    maxwidths=[400, 400, 400], wide=False,
                                    writeout="small_names.txt"):
        f = codecs.open(csv_path, encoding='utf-8', mode='r+')
        self.data = [row.split('\t') for row in f]
        self.header = self.data[0]
        self.data = self.data[1:]

        self.root_path = root_path
        self.img_height = img_height
        self.char_to_idx = char_to_idx
        self.augmentation = augmentation
        self.categories = categories # ['birthplace', 'fthr_birthplace', 'mthr_birthplace']
        self.maxwidths = maxwidths
        self.wide = wide
        self.writeout = writeout

    def __len__(self):
        return len(self.data) * len(self.categories)

    def __getitem__(self, idx):
        sub = int(idx/len(self.data))
        item = self.data[(idx%len(self.data))]

        img = cv2.imread(os.path.join(self.root_path, item[0]) + '.jpg')

        if img is None:
            print("Warning: image is None:", os.path.join(self.root_path, item[0]))
            return None

        xy = item[self.categories[sub]+1:self.categories[sub]+5]
        # min_x, max_x, min_y, max_y = item[self.categories[sub]+1:self.categories[sub]+5]
        min_x, max_x, min_y, max_y = int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])
        if self.wide:
            min_y -= 25
            max_y += 25
            if max_y > img.shape[0]:
                max_y = img.shape[0]
        img = img[min_y:max_y, min_x:max_x]

        if img.shape[0] < 1:
            print("Warning: segment is None:", os.path.join(self.root_path, item[0]))
            return None

        percent = float(self.img_height) / img.shape[0]
        img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)

        if self.augmentation:
            img = grid_distortion.warp_image(img)

        img = img.astype(np.float32)
        img = img / 128.0 - 1.0

        if img.shape[1] > self.maxwidths[sub]:
            # print(idx, item, (max_y - min_y), img.shape[1])
            with open(self.writeout, 'a') as f:
                f.write(str(idx))
                f.write(',')
                f.write(str(item))
                f.write(',')
                f.write(str(max_y - min_y))
                f.write('\n')
            return None

        gt = item[self.categories[sub]]
        if gt == ' ':
            gt = '@'
        gt_label = string_utils.str2label(gt, self.char_to_idx)

        return {
            "line_img": img,
            "gt_label": gt_label,
            "gt": gt,
            "idx": idx,
            "item": item
        }
