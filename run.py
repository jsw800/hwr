import json

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import hw_dataset
from hw_dataset import HwDataset

from torch.utils.data import DataLoader
from warpctc_pytorch import CTCLoss
import error_rates
import string_utils

import crnn

import character_set
import os
import sys
import cv2
import numpy as np
import codecs

import random
import string_utils

def main():
  config_path = sys.argv[1]

  with open(config_path) as f:
    config = json.load(f)

  idx_to_char, char_to_idx = character_set.load_char_set(config['character_set_path'])

  hw = crnn.create_model({
    'cnn_out_size': config['network']['cnn_out_size'],
    'num_of_channels': 3,
    'num_of_outputs': len(idx_to_char)+1
  })


  hw.load_state_dict(torch.load(config['model_save_path']))
  if torch.cuda.is_available():
    hw.cuda()
    dtype = torch.cuda.FloatTensor
    # print("Using GPU")
  else:
    dtype = torch.FloatTensor
    # print("No GPU detected")

  hw.eval()

  val_dataset = HwDataset(config["validation_set_path"], char_to_idx,
    img_height=config['network']['input_height'],
    root_path=config['image_root_directory'], augmentation=True,
    categories=config['categories'], maxwidths=config['maxwidths'],
    wide=config['wide'], writeout=config['error_log']
    )

  total = 0
  incorrect = 0
  f = codecs.open(config["incorrect_save_path"], encoding='utf-8', mode='w+')
  for i, im in enumerate(val_dataset):
    if im is not None:
      total += 1
      label = im["gt"]

      img = im['line_img']
      img = np.expand_dims(img, axis=0)
      img = img.transpose([0,3,1,2])
      img = Variable(torch.from_numpy(img).type(torch.cuda.FloatTensor),
          requires_grad=False)
      preds = hw(img)

      output_batch = preds.permute(1,0,2)
      out = output_batch.data.cpu().numpy()

      pred, pred_raw = string_utils.naive_decode(out[0])
      pred_str = string_utils.label2str(pred, idx_to_char, False)
      print(pred_str)
      pred_raw_str = string_utils.label2str(pred_raw, idx_to_char, True)
      if pred_str.replace(' ', '').replace(',', '') != \
            label.replace(' ', '').replace(',', ''):
        incorrect += 1
        try:
          f.write("index: {}, label: {}, prediction: {}\n".format(
            im['idx'], label, pred_str))
        except Exception as e:
          continue
      if i % 1000 == 999:
        print("Total: {}, Incorrect: {}, Accuracy: {}".format(
          total, incorrect, 1.0 - incorrect/float(total)))
        with open(config["accuracy_path"], "w") as f2:
          f2.write("Total: {}, Incorrect: {}, Accuracy: {}\n".format(
              total, incorrect, 1.0 - incorrect/float(total)))

  print("Total: {}, Incorrect: {}, Accuracy: {}".format(
    total, incorrect, 1.0 - incorrect/float(total)))
  with open(config["accuracy_path"], "w") as f2:
    f2.write("Total: {}, Incorrect: {}, Accuracy: {}\n".format(
      total, incorrect, 1.0 - incorrect/float(total)))

  return 1.0 - incorrect/float(total)

if __name__ == "__main__":
  main()
