import load_modules
import sys
from datasets.Dataset import HwDataset


def run(modules_config, image_folder, segmentation_path, output_filename):
    with open(segmentation_path) as f:
        segmentation = [row.split('\t') for row in f.read().split('\n') if row != ''][1:]
    modules = load_modules.load(modules_config)
    dataset = HwDataset(image_folder, segmentation)
    output = {}
    prev_img_name = None
    line_number = 1
    for item in dataset:
        if prev_img_name is None or item['image_name'] != prev_img_name:
            prev_img_name = item['image_name']
            line_number = 1
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
        output[line_id] = line_output
        print(line_number)
        line_number += 1

    # We now have the output for this data folder, output it.
    with open(output_filename, "w+") as f:
        for i, module in enumerate(modules):
            f.write(module['field_name'])
            if i != len(modules) - 1:
                f.write('\t')
        f.write('\n')
        for key in output:
            line_output = output[key]
            out_str = ''
            for i, module in enumerate(modules):
                out_str += line_output[module['field_name']]
                if i != len(modules) - 1:
                    out_str += '\t'
            out_str += '\n'
            f.write(out_str)


if __name__ == '__main__':
    run('configs/none_config.yaml', 'data/images/004955523', 'data/segmentation/004955523.csv',
                    'output/004955523.csv')
