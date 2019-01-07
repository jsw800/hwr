import load_modules
import sys
from datasets.Dataset import HwDataset
from printer import Printer


def run(modules_config, image_folder, segmentation_path, output_filename):
    with open(segmentation_path) as f:
        segmentation = [row.split('\t') for row in f.read().split('\n') if row != '']
    segmentation = segmentation[1:]
    modules = load_modules.load(modules_config)
    printer = Printer(output_filename, [module['field_name'] for module in modules])
    printer.write_header()
    dataset = HwDataset(image_folder, segmentation)
    prev_img_name = None
    count = 0
    line_number = 1
    for item in dataset:
        if prev_img_name is None or item['image_name'] != prev_img_name:
            prev_img_name = item['image_name']
            line_number = 1
            count += 1
            # print number of images read so far
            print(count)
        img_name = item['image_name']
        images = item['fields']
        line_output = {}

        if len(images) != len(modules):
            sys.stderr.write("number of fields != expected number of fields for image " + img_name +
                                                                    "line # " + str(line_number) + "\n")
        for i in range(min(len(images), len(modules))):
            module = modules[i]
            image = images[i]
            pred = module['recognition'].run(image)
            corrected_pred = module['postprocessing'].postprocess(pred)
            line_output[module['field_name']] = corrected_pred

        line_print = str(line_number)
        line_print = line_print if len(line_print) == 2 else '0' + line_print
        line_id = img_name.split('/')[-1].split('.')[0] + '_' + line_print
        printer.write_line(line_id, line_output)
        line_number += 1

    printer.close()


if __name__ == '__main__':
    run('configs/config.yaml', 'data/images/004955523', 'data/segmentation/004955523.csv',
                    'output/004955523.csv')
