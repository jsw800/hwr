import yaml
from recognition.none import NoneRecognitionModule
from recognition.crnn import CRNNRecognition
from recognition.classifier import ClassifierRecognition
from postprocessing.none import NonePostProcessModule
from postprocessing.dictionary import DictionaryPostProcess

# REGISTER MODULE NAMES HERE, add new modules here when you want to use them

RECOGNITION_MODULE_NAMES = {
    'none': NoneRecognitionModule,
    'crnn': CRNNRecognition,
    'classifier': ClassifierRecognition
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
    for field_config in config:
        field_names.append(field_config['field_name'])
        field_pretty_names.append(field_config['pretty_field_name'])

        # get class names
        rec_class = RECOGNITION_MODULE_NAMES[field_config['method']]
        postproc_class = POSTPROCESS_MODULE_NAMES[field_config['postprocessing']]

        # construct rec/postproc objects with args from config
        rec_module = rec_class(tuple(field_config['recognition_args']))
        postproc_module = postproc_class(tuple(field_config['postprocessing_args']))

        recognition_modules.append(rec_module)
        postprocess_modules.append(postproc_module)

    return [{
        'field_name': field_names[i],
        'field_pretty_name': field_pretty_names[i],
        'recognition': recognition_modules[i],
        'postprocessing': postprocess_modules[i]
    } for i in range(len(config))]


