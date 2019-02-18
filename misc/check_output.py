import sys
import os
from os.path import join
from PIL import Image
import random


def load_csv(path):
    with open(path) as f:
        retval = [row.split('\t') for row in f.read().split('\n')]
    return retval


output_path = sys.argv[1]
segmentation_path = sys.argv[2]
image_folder = sys.argv[3]
field_name = sys.argv[4]

output = load_csv(output_path)
fields = output[0]
output = output[1:]
# random sample output
random.shuffle(output)

segs = load_csv(segmentation_path)
segs_fields = segs[0]
segs = segs[1:]

if field_name not in fields or field_name not in segs_fields:
    sys.stderr.write("Field not found\n")
    exit(1)

field_idx = fields.index(field_name)
segs_idx = segs_fields.index(field_name)

for row in output:
    idnt = row[0]
    img_name = '_'.join(idnt.split('_')[:2])
    row_num = int(idnt.split('_')[2]) - 1       # 1-indexed
    these_segs = [seg for seg in segs if seg[0] == img_name]
    seg = these_segs[row_num]
    xmin, xmax, ymin, ymax = seg[segs_idx], seg[segs_idx + 1], seg[segs_idx + 2], seg[segs_idx + 3]
    img_path = join(image_folder, img_name + '.jpg')
    img = Image.open(img_path)
    coords = (float(xmin), float(ymin), float(xmax), float(ymax))
    cell = img.crop(coords)
    cell.show()
    print(row[field_idx])
    raw_input()
    os.system('killall display')

