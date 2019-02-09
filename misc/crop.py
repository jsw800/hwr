from PIL import Image
import sys
from os.path import join

with open(sys.argv[1]) as f:
    data = [row.split('\t') for row in f.read().split('\n') if 'veteran' not in row]

directory = sys.argv[1].split('.')[0]


cur_image_name = ''

for i, row in enumerate(data):
    image_name = directory + '/' + row[0] + '.jpg'
    if cur_image_name != image_name:
        idx = 0
        img = Image.open(image_name)
        cur_image_name = image_name
    if int(row[105]) - int(row[106]) == 0 or int(row[107]) - int(row[108]) == 0:
        idx += 1
        continue
    coords = (float(row[105]), float(row[107]), float(row[106]), float(row[108]))
    print(row[0] + '.jpg')
    print(coords)
    print(img.size)
    new_image = img.crop(coords)
    idx_str = str(idx)
    if len(idx_str) == 1:
        idx_str = '0' + idx_str
    new_image.save('war_images/' + row[0] + "_" + idx_str + '.jpg')
    idx += 1
    if i > 80000:
        break
