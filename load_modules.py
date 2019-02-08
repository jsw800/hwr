import yaml
from recognition.none import NoneRecognitionModule
from recognition.crnn import CRNNRecognition
from recognition.classifier import ClassifierRecognition
from postprocessing.none import NonePostProcessModule
from postprocessing.dictionary import DictionaryPostProcess
from postprocessing.lastnamecorrection import LastNameCorrection
from preprocessing.none import NonePreprocessModule

# REGISTER MODULE NAMES HERE, add new modules here when you want to use them

PREPROCESS_MODULE_NAMES = {
    'none': NonePreprocessModule
}

RECOGNITION_MODULE_NAMES = {
    'none': NoneRecognitionModule,
    'crnn': CRNNRecognition,
    'classifier': ClassifierRecognition
}

POSTPROCESS_MODULE_NAMES = {
    'none': NonePostProcessModule,
    'dictionary': DictionaryPostProcess,
    'lastnamecorrection': LastNameCorrection
}


def load(config_path):

    with open(config_path) as f:
        config = yaml.load(f)

    field_names = []
    field_pretty_names = []
    preprocess_modules = []
    recognition_modules = []
    postprocess_modules = []
    for field_config in config:
        field_names.append(field_config['field_name'])
        field_pretty_names.append(field_config['pretty_field_name'])

        # get class names
        preproc_class = PREPROCESS_MODULE_NAMES[field_config['preprocessing']]
        rec_class = RECOGNITION_MODULE_NAMES[field_config['method']]
        postproc_class = POSTPROCESS_MODULE_NAMES[field_config['postprocessing']]

        # construct rec/postproc objects with args from config
        preproc_module = preproc_class(tuple(field_config['preprocessing_args']))
        rec_module = rec_class(tuple(field_config['recognition_args']))
        postproc_module = postproc_class(tuple(field_config['postprocessing_args']))

        preprocess_modules.append(preproc_module)
        recognition_modules.append(rec_module)
        postprocess_modules.append(postproc_module)

    return [{
        'field_name': field_names[i],
        'field_pretty_name': field_pretty_names[i],
        'preprocessing': preprocess_modules[i],
        'recognition': recognition_modules[i],
        'postprocessing': postprocess_modules[i]
    } for i in range(len(config))]


