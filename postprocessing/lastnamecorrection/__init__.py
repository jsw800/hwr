from postprocess_module import PostprocessModule
import editdistance as ed
from os.path import join

THIS_DIR_PATH = 'postprocessing/lastnamecorrection'

class LastNameCorrection(PostprocessModule):

    def __init__(self, args):
        dict_path = args[0]
	with open(join(THIS_DIR_PATH, dict_path)) as f:
                self.dictionary = list(set([row for row in f.read().split('\n') if row != '']))

    def postprocess(self, recognizer_output): 
	if recognizer_output[0] == '-':
	    return self.previous_output
	else:
	    self.previous_output = recognizer_output
	    return recognizer_output
