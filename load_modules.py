import yaml
from recognition.none import NoneRecognitionModule
from recognition.crnn import CRNNRecognition
from postprocessing.none import NonePostProcessModule
from postprocessing.dictionary import DictionaryPostProcess

# MAP MODULE NAMES HERE, add new modules here when you want to use them

RECOGNITION_MODULE_NAMES = {
    'none': NoneRecognitionModule,
    'crnn': CRNNRecognition
}

POSTPROCESS_MODULE_NAMES = {
    'none': NonePostProcessModule,
    'dictionary': DictionaryPostProcess
}


def load(config_path):

    with open(config_path) as f:
        config = yaml.load(f)

    field_names = []
    field_pretty_names = []
    recognition_modules = []
    postprocess_modules = []
    for module in config:
        field_names.append(module['field_name'])
        field_pretty_names.append(module['pretty_field_name'])
        recognition_modules.append(
            RECOGNITION_MODULE_NAMES[module['method']](tuple(module['recognition_args'])))
        postprocess_modules.append(
            POSTPROCESS_MODULE_NAMES[module['postprocessing']](tuple(module['postprocess_args'])))

    return [{
        'field_name': field_names[i],
        'field_pretty_name': field_pretty_names[i],
        'recognition': recognition_modules[i],
        'postprocessing': postprocess_modules[i]
    } for i in range(len(config))]


if __name__ == "__main__":
    x = load('configs/none_config.yaml')
    print(x)
