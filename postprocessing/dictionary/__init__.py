from postprocess_module import PostprocessModule
import editdistance as ed
from os.path import join

# TODO: upper limit on editdistance? If the label is bad enough, we might want to reject and say unreadble


THIS_DIR_PATH = 'postprocessing/dictionary'


class DictionaryPostProcess(PostprocessModule):

    def __init__(self, args):
        dict_path = args[0]
        with open(join(THIS_DIR_PATH, dict_path)) as f:
            self.dictionary = set([row for row in f.read().split('\n') if row != ''])

    def postprocess(self, recognizer_output):
        if recognizer_output in self.dictionary:
            return recognizer_output
        min_ed = float('inf')
        best = None
        for label in self.dictionary:
            dist = ed.eval(label.lower(), recognizer_output.lower())
            if dist < min_ed:
                min_ed = dist
                best = label

        return best

