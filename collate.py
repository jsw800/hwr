import torch
import numpy as np
import cv2

PADDING_CONSTANT = 0

def collate_field(batch):
    batch = [b for b in batch if b is not None]
    maxheight = max([b.shape[0] for b in batch])
    for i, b in enumerate(batch):
        percent = float(maxheight) / b.shape[0]
        batch[i] = cv2.resize(b, (0,0), fx=percent, fy=percent, interpolation=cv2.INTER_CUBIC)
    assert len(set([b.shape[0] for b in batch])) == 1
    assert len(set([b.shape[2] for b in batch])) == 1

    dim0 = batch[0].shape[0]
    dim1 = max([b.shape[1] for b in batch])
    dim2 = batch[0].shape[2]

    input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)
    for i in xrange(len(batch)):
        b_img = batch[i]
        input_batch[i,:,:b_img.shape[1],:] = b_img

#??    line_imgs = input_batch.transpose([0,3,1,2])  # I think this is fixing the cv2 imread dims

    return input_batch


def collate(batch):
    lines = [line['fields'] for line in batch]
    imgs = [line['image_name'] for line in batch]
    by_field = [[line[i] for line in lines] for i in range(len(lines[0]))]
    out = []
    for field in by_field:
        out.append(collate_field(field))
    return {
        'fields':out,
        'img_names': imgs
    }
