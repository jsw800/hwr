from PIL import Image
import sys
import os
from os.path import join

os.system("touch labels.csv")

with open('labels.csv') as f:
    contents = [row.split('\t') for row in f.read().split('\n')]

already_labelled = [row[0] for row in contents]
print(len(already_labelled))

folder_name = sys.argv[1]
images = []
for root, dirs, files in os.walk(folder_name):
    images += files

images = sorted([join(folder_name, img) for img in images if img not in already_labelled])

for im_name in images:
    im_basic_name = im_name.split('/')[-1]
    im = Image.open(im_name)
    im.show()
    label = raw_input("Label: ")
    im.close()
    with open('labels.csv', 'a+') as f:
       f.write(im_basic_name + '\t' + label + '\n')
    os.system('killall display')
