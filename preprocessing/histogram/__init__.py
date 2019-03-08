from preprocess_module import PreprocessModule
import cv2
import numpy as np


class HistogramEqualization(PreprocessModule):

    # no args
    def __init__(self, args):
        pass

    def preprocess(self, input_image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
        out = np.zeros_like(input_image)
        for i in range(input_image.shape[2]):
            a = input_image[:, :, i].astype(np.uint8)
            out[:, :, i] = clahe.apply(a)

        return out

    def batch_preprocess(self, input_image):
        out = np.zeros_like(input_image)
        for i in range(input_image.shape[0]):
            out[i] = self.preprocess(input_image[i])
        return out
