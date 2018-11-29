from os import system
import json
import random

with open("PR_BIRTHPLACE/validation.json") as f:
    imgs = f.read()

imgs = json.loads(imgs)

random.shuffle(imgs)

for img in imgs:
    print(img["image_path"])
    system("python recognize.py PR_BIRTHPLACE/config.json PR_BIRTHPLACE/PR_BIRTHPLACE/" + img["image_path"])
    print(img["gt"])
    raw_input()
