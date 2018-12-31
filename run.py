import load_modules
import sys
import os
import cv2
from datasets.Dataset import HwDataset


def run(modules_config, image_folder, segmentation_path):
    with open(segmentation_path) as f:
        segmentation = [row.split('\t') for row in f.read().split('\n')][1:]
    modules = load_modules.load(modules_config)
    dataset = HwDataset(image_folder, segmentation)
    out = dataset[0]


if __name__ == '__main__':
    run('configs/none_config.yaml', 'data/images/004955523', 'data/segmentation/004955523.csv')
