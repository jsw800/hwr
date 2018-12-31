from postprocess_module import PostprocessModule


class NonePostProcessModule(PostprocessModule):

    # no args for none modules
    def __init__(self, args):
        pass

    def postprocess(self, recognizer_output):
        return recognizer_output
