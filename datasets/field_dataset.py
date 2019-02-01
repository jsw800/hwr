from torch.utils.data import Dataset
from os.path import join
import cv2


class FieldDataset(Dataset):

    # segmentation should include headers, with field names matching the modules config field names.
    def __init__(self, image_folder, segmentation, field_name):
        headers = segmentation[0]

        # get index of the field we want
        for i, header in enumerate(headers):
            if header == field_name:
                self.which = i
                break

        self.segmentation = segmentation[1:]
        self.image_folder = image_folder
        self.current_image_name = None
        self.current_image = None

    def __len__(self):
        return len(self.segmentation)

    def __getitem__(self, idx):
        segs = self.segmentation[idx]
        img_name = segs[0]
        segs = segs[self.which:self.which + 4]
        segs = [int(coord) for coord in segs]
        if join(self.image_folder, img_name) != self.current_image_name:
            self.current_image_name = join(self.image_folder, img_name)
            self.current_image = cv2.imread(self.current_image_name + '.jpg')
        return {
            'image_name': self.current_image_name + '.jpg',
            'img': self.current_image[segs[2]:segs[3], segs[0]:segs[1]]
        }


