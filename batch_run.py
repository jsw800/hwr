import load_modules
import sys
from datasets import FullDataset
from printer import Printer
from torch.utils.data import DataLoader
from collate import collate

def run(modules_config, image_folder, segmentation_path, output_filename):

    # Load all objects we will need throughout the run process
    # (segmentation data, recognition/postprocess modules, printer, and dataset (image loader))
    with open(segmentation_path) as f:
        segmentation = [row.split('\t') for row in f.read().split('\n') if row != '']
    segmentation = segmentation[1:]
    modules = load_modules.load(modules_config)
    printer = Printer(output_filename, [module['field_name'] for module in modules])
    printer.write_header()
    dataset = FullDataset(image_folder, segmentation)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate)

    prev_img_name = None
    count = 0
    line_number = 1

    for i, batch in enumerate(dataloader):
        img_names = batch['img_names']
        if prev_img_name is None:
            prev_img_name = img_names[0]
            line_number = 1
        images = batch['fields']
        line_output = {}

        # warn possible segmentation errors, but don't auto fail, just warn
        if len(images) != len(modules):
            sys.stderr.write("number of fields != expected number of fields for images " +
                        img_name + "line # " + str(line_number) + "\n")

        # Read each field on this line into line_output
        for i in range(min(len(images), len(modules))):
            module = modules[i]
            image = images[i]
            rectified_img = module['preprocessing'].batch_preprocess(image)
            pred = module['recognition'].batch_run(rectified_img)
            corrected_pred = module['postprocessing'].batch_postprocess(pred)
            line_output[module['field_name']] = corrected_pred

        # output this batch's labels
        line_ids = []
        for i, img_nm in enumerate(img_names):
            if img_nm != prev_img_name:
                print(prev_img_name)
                print(img_nm)
                prev_img_name = img_nm
                line_number = 1
                count += 1
                # log number of pages read
                print(count)
            lnnm = str(line_number)
            lnnm = lnnm if len(lnnm) == 2 else '0' + lnnm
            idnt = img_nm.split('/')[-1].split('.')[0] + '_' + lnnm
            line_ids.append(idnt)
            line_number += 1
        printer.write_batch(line_ids, line_output)
        line_number += 1

    printer.close()


if __name__ == '__main__':
    run('configs/config.yaml', 'data/images/004953237', 'data/segmentation/004953237.csv',
                    'output/004953237.csv')
