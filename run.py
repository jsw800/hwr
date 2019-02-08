import load_modules
import sys
from datasets import FullDataset
from printer import Printer


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

    prev_img_name = None
    count = 0
    line_number = 1

    for census_line in dataset:
        if prev_img_name is None or census_line['image_name'] != prev_img_name:
            prev_img_name = census_line['image_name']
            line_number = 1
            count += 1
            # print number of census pages read so far
            print(count)
        img_name = census_line['image_name']
        images = census_line['fields']
        line_output = {}

        # warn possible segmentation errors, but don't auto fail, just warn
        if len(images) != len(modules):
            sys.stderr.write("number of fields != expected number of fields for image " + img_name +
                                                                    "line # " + str(line_number) + "\n")

        # Read each field on this line into line_output
        for i in range(min(len(images), len(modules))):
            module = modules[i]
            image = images[i]
            rectified_img = module['preprocessing'].preprocess(image)
            pred = module['recognition'].run(rectified_img)
            corrected_pred = module['postprocessing'].postprocess(pred)
            line_output[module['field_name']] = corrected_pred

        # output this line's labels
        line_print = str(line_number)
        line_print = line_print if len(line_print) == 2 else '0' + line_print
        line_id = img_name.split('/')[-1].split('.')[0] + '_' + line_print
        printer.write_line(line_id, line_output)
        line_number += 1

    printer.close()


if __name__ == '__main__':
    run('configs/config.yaml', 'data/images/004949552', 'data/segmentation/004949552.csv',
                    'output/004949552.csv')
