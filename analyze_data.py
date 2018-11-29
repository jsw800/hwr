import json
import character_set
import sys
import hw_dataset
from hw_dataset import HwDataset
import crnn
import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss
import error_rates
import string_utils
import cv2

# config_path = "configs/wide_names&birth_config.json"
config_path = sys.argv[1]

with open(config_path) as f:
    config = json.load(f)

idx_to_char, char_to_idx = character_set.load_char_set(config['character_set_path'])

data = HwDataset(config['validation_set_path'], char_to_idx,
    img_height=config['network']['input_height'], root_path=config['image_root_directory'],
    categories=config['categories'], maxwidths=config['maxwidths'], wide=config['wide'])


def get_avgs():
    xsu = 0
    for i in range(100):
        xsu += data[i]['line_img'].shape[1]

    avg = float(xsu) / 100
    print('avg width: {}'.format(float(xsu) / 100))

    print(len(data))
    start = len(data) / 2
    print(start)
    xsu = 0
    for i in range(start,start+100):
        xsu += data[i]['line_img'].shape[1]

    avg = float(xsu) / 100
    print('avg width: {}'.format(float(xsu) / 100))

def segment_show(idx):
    cv2.imwrite(data[idx]['line_img'], 'showing.jpg')

get_avgs()
