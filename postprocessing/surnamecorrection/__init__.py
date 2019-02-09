from postprocess_module import PostprocessModule
import editdistance as ed
from os.path import join

THIS_DIR_PATH = 'postprocessing/surnamecorrection'

class SurnameCorrection(PostprocessModule):

    def __init__(self, args):
        self.previous_output = ""

    def postprocess(self, recognizer_output): 
	if recognizer_output[0] == '-':
	    return self.previous_output
	else:
	    self.previous_output = recognizer_output
	    return recognizer_output
