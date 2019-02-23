from recognition_module import RecognitionModule


DEFAULT_OUTPUT = "--"


class NoneRecognitionModule(RecognitionModule):

    # no args for none modules
    def __init__(self, args):
        pass

    def run(self, input_image):
        return DEFAULT_OUTPUT

    def batch_run(self, input_batch):
        return [DEFAULT_OUTPUT for i in range(input_batch.shape[0])]
