import json

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

from collections import defaultdict
import os
import cv2
import numpy as np

import random

import grid_distortion

PADDING_CONSTANT = 0

# This creates the batches and zero pads the images that are smaller than the longest one,
# so every one in the batch is the same size. line_imgs is a 4d tensor
# with shape (batch_size, num_channels, img_height, img_width)
def collate(batch):
    batch = [b for b in batch if b is not None]
    #These all should be the same size or error
    assert len(set([b['line_img'].shape[0] for b in batch])) == 1
    assert len(set([b['line_img'].shape[2] for b in batch])) == 1

    # dim0 = img_height
    dim0 = batch[0]['line_img'].shape[0]
    # dim1 = img_width
    dim1 = max([b['line_img'].shape[1] for b in batch])
    # dim2 = num of color channels
    dim2 = batch[0]['line_img'].shape[2]
    

    # zero pad small images
    input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)

    for i in xrange(len(batch)):
        b_img = batch[i]['line_img']
        input_batch[i,:,:b_img.shape[1],:] = b_img

    # (batch_size, num_color_channels, img_height, img_width)
    line_imgs = input_batch.transpose([0,3,1,2])
    line_imgs = torch.from_numpy(line_imgs)

    return {
        "line_imgs": line_imgs,
        "gt": [b['gt'] for b in batch]
    }

class HwDataset(Dataset):
    def __init__(self, data, img_height=32, augmentation=False):

        self.data = data
        self.img_height = img_height
        self.augmentation = augmentation

    # enables len(dataset)
    def __len__(self):
        return len(self.data)

    # This function enables subscripting ( dataset[idx] )
    # when the dataset object is subscripted, this determines what is returned.
    def __getitem__(self, idx):
        # get data for this idx
        item = self.data[idx]

        # read the image
        img = cv2.imread(item['im'])

        if img is None:
            print("Warning: image is None:", item['im'])
            return None

        # resize the image (in this net, we want square images)
        percent = float(self.img_height) / img.shape[0]
        #img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)
        img = cv2.resize(img, (self.img_height, self.img_height), interpolation=cv2.INTER_CUBIC)

        # image augmentation (this basically distorts the image so we can have more training data
        # without hand-generating it)
        if self.augmentation:
            img = grid_distortion.warp_image(img) 

        img = img.astype(np.float32)
        img = img / 128.0 - 1.0

        gt = item['gt']

        # return a dict object with these elements
        return {
            "line_img": img,
            "gt": gt,
            'filename' : item['im']
        }
