import json

import torch
from torch.autograd import Variable

import cv2
import numpy as np
import cnn as model
from recognition_module import RecognitionModule
from os.path import join

THIS_DIR_PATH = 'recognition/classifier'

class ClassifierRecognition(RecognitionModule):
    
    # args should have the config path only.
    def __init__(self, args):

        config_path = join(THIS_DIR_PATH, args[0])
        with open(config_path) as f:
            self.config = json.load(f)
        self.img_height = self.config['network']['input_height']
        self.network = model.create_model(self.config)
        if torch.cuda.is_available():
            self.network.cuda()
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor


    def run(self, input_image):
        input_image = cv2.resize(input_image, (self.img_height, self.img_height),
                                                    interpolation=cv2.INTER_CUBIC)
        input_image = input_image.astype(np.float32)
        
        preds = self.network(input_image).cpu().numpy()
        pred = np.argmax(preds)
        for key in self.config['classes']:
            if self.config['classes'][key] == pred:
                return key
        return "CLASSIFICATION ERROR"

