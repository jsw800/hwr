from preprocess_module import PreprocessModule


class NonePreprocessModule(PreprocessModule):

    def __init__(self, args):
        pass

    def preprocess(self, input_image):
        return input_image

    def batch_preprocess(self, input_batch):
        return input_batch
