import json

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

import model

import os
import sys
import cv2
import numpy as np

import random

def main():
    # load config
    config_path = sys.argv[1]
    image_path = sys.argv[2]

    with open(config_path) as f:
        config = json.load(f)

    # initialize the net
    hw = model.create_model(config)

    # load existing weights file
    hw.load_state_dict(torch.load(config['model_save_path']))

    # Check if we can use the GPU
    if torch.cuda.is_available():
        hw.cuda()
        dtype = torch.cuda.FloatTensor
        # print("Using GPU")
    else:
        dtype = torch.FloatTensor
        # print("No GPU detected")

    # put the net in eval mode
    hw.eval()

    # read the image from the file path
    img = cv2.imread(image_path)
    # if ig doesn't exist, give up
    if img is None:
        print("Image not found")
        exit()

    # resize the image (We are using square images of the same size each time)
    img_height = config['network']['input_height']
    img = cv2.resize(img, (img_height, img_height), interpolation = cv2.INTER_CUBIC)

    # Prepare for net
    img = torch.from_numpy(img.transpose(2,0,1).astype(np.float32)/128 - 1)
    img = Variable(img[None,...].type(dtype), requires_grad=False, volatile=True)

    # put it through the net
    preds = hw(img)
    # interpret output
    preds = preds.data.cpu().numpy()
    pred = np.argmax(preds)
    # log output
    if pred == 0:
        print('not WW')
    if pred == 1:
        print('WW')


if __name__ == "__main__":
    main()
