from torch.utils.data import Dataset
from os.path import join
import cv2


class HwDataset(Dataset):

    def __init__(self, image_folder, segmentation):
        self.segmentation = segmentation
        self.image_folder = image_folder
        self.current_image_name = None
        self.current_image = None

    def __len__(self):
        return len(self.segmentation)

    # return a list of images, one for each field
    def __getitem__(self, item):
        retval = []
        segs = self.segmentation[item]
        # cache currently open image to save time opening
        if join(self.image_folder, segs[0]) + '.jpg' != self.current_image_name:
            self.current_image_name = join(self.image_folder, segs[0]) + '.jpg'
            self.current_image = cv2.imread(self.current_image_name)
        segs = segs[1:]
        segs = [int(coord) for coord in segs]
        for i in range(0, len(segs), 4):
            retval.append(self.current_image[segs[i+2]:segs[i+3], segs[i]:segs[i+1]])
        return {
            'image_name': self.current_image_name,
            'fields': retval
        }
