from recognition_module import RecognitionModule
import json
import crnn
import torch
import cv2
import numpy as np
from torch.autograd import Variable
import string_utils
import character_set
from os.path import join


THIS_DIR_PATH = "recognition/crnn"


class CRNNRecognition(RecognitionModule):

    # One argument - name of config file
    def __init__(self, args):
        config_path = args[0]
        with open(join(THIS_DIR_PATH, config_path)) as f:
            self.config = json.load(f)
        self.idx_to_char, self.char_to_idx = character_set.load_char_set(join(THIS_DIR_PATH,
                                                                              self.config['character_set_path']))
        self.network = crnn.create_model({
            'cnn_out_size': self.config['network']['cnn_out_size'],
            'num_of_channels': 3,
            'num_of_outputs': len(self.idx_to_char) + 1
        })
        if torch.cuda.is_available():
            self.network.load_state_dict(torch.load(join(THIS_DIR_PATH, self.config['model_save_path'])))
        else:
            self.network.load_state_dict(torch.load(join(THIS_DIR_PATH, self.config['model_save_path']), map_location='cpu'))
        if torch.cuda.is_available():
            self.network.cuda()
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        self.network.eval()

    def run(self, img):
        if img.shape[0] != self.config['network']['input_height']:
            percent = float(self.config['network']['input_height']) / img.shape[0]
            img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation=cv2.INTER_CUBIC)

        img = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 128 - 1)
        img = Variable(img[None, ...].type(self.dtype), requires_grad=False, volatile=True)

        try:
            preds = self.network(img)
        except Exception as e:
            print(e)
            return "UNREADABLE"

        output_batch = preds.permute(1, 0, 2)
        out = output_batch.data.cpu().numpy()

        pred, pred_raw = string_utils.naive_decode(out[0])
        pred_str = string_utils.label2str(pred, self.idx_to_char, False)
        pred_raw_str = string_utils.label2str(pred_raw, self.idx_to_char, True)
        return pred_str

    def batch_run(self, input_batch):
        imgs = torch.from_numpy(input_batch.astype(np.int32)).type(self.dtype)
        line_imgs = Variable(imgs, requires_grad=False, volatile=True)
        try:
            preds = self.network(img)
        except Exception as e:
            print("AAAAH???")
            return []
        output_batch = preds.permute(1, 0, 2)
        out = output_batch.data.cpu().numpy()

        retval = []
        for i in range(out.shape[0]):
            logits = out[i, ...]
            pred, predPraw = string_utils.naive_decode(logits)
            pred_str = string_utils.label2str(pred, self.idx_to_char, False)
            print(pred_str)
            retval.append(pred_str)
        return retval
