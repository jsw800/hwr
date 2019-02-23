import json

import torch
from torch.autograd import Variable

import cv2
import numpy as np
import cnn as model
from recognition_module import RecognitionModule
from os.path import join

THIS_DIR_PATH = 'recognition/classifier'

def batch_resize(batch, height):
    a = cv2.resize(batch[0], (height, height), interpolation=cv2.INTER_CUBIC)
    out = np.zeros((8, a.shape[0], a.shape[1], a.shape[2]))
    out[0] = a 
    for i in range(1, 8): 
        out[i] = cv2.resize(batch[i], (height, height), interpolation=cv2.INTER_CUBIC)
    return out

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
            self.network.load_state_dict(torch.load(join(THIS_DIR_PATH, self.config['model_save_path'])))
        else:
            self.dtype = torch.FloatTensor
            self.network.load_state_dict(torch.load(join(THIS_DIR_PATH, self.config['model_save_path']),
                                                    map_location='cpu'))
        self.network.eval()


    def run(self, input_image):
        input_image = cv2.resize(input_image, (self.img_height, self.img_height),
                                                    interpolation=cv2.INTER_CUBIC)
        input_image = input_image.astype(np.float32)
        input_image = input_image / 128.0 - 1.0
        input_image = torch.from_numpy(input_image.transpose(2,0,1))
        input_image = Variable(input_image[None, ...].type(self.dtype), requires_grad=False, volatile=True)
        
        preds = self.network(input_image).data.cpu().numpy()
        pred = np.argmax(preds)
        for key in self.config['classes']:
            if self.config['classes'][key] == pred:
                return key
        return "CLASSIFICATION ERROR"

    def batch_run(self, input_batch):
        input_batch = batch_resize(input_batch, self.img_height)
        input_batch = input_batch.astype(np.float32)
        input_batch = input_batch / 128.0 - 1.0
        input_batch = torch.from_numpy(input_batch.transpose([0,3,1,2])).type(self.dtype)
        input_batch = Variable(input_batch, requires_grad=False, volatile=True)

        preds = self.network(input_batch).data.cpu().numpy()
        preds = np.argmax(preds, axis=1)
        out = []
        # TODO: handle errors?
        for pred in preds:
            for key in self.config['classes']:
                if self.config['classes'][key] == pred:
                    out.append(key)
                    break
        return out
