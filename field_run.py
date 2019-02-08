import load_modules
import sys
from datasets import FieldDataset

def run(modules_config, image_folder, segmentation_path, field_name):
    
    with open(segmentation_path) as f:
        segmentation = [row.split('\t') for row in f.read().split('\n') if row != '']
    # TODO: make specific field load function, we are constructing a bunch of modules
    # that are just being thrown away here
    modules = load_modules.load(modules_config)
    for m in modules:
        if m['field_name'] == field_name:
            module = m
            modules = None
            break
    dataset = FieldDataset(image_folder, segmentation, field_name)
    prev_img_name = None
    for census_line in dataset:
        img_name = census_line['image_name']
        img = census_line['img']
        rectified_img = module['preprocessing'].preprocess(img)
        pred = module['recognition'].run(img)
        corrected_pred = module['postprocessing'].postprocess(pred)
        print(img_name + '\t' + corrected_pred)


if __name__ == "__main__":
    run('configs/config.yaml', 'data/images/004949552', 'data/segmentation/004949552.csv',
                sys.argv[1])
