from postprocess_module import PostprocessModule
import editdistance as ed
from os.path import join

THIS_DIR_PATH = 'postprocessing/surnamecorrection'

class SurnameCorrection(PostprocessModule):

    def __init__(self, args):
        self.previous_surname = ""

    def postprocess(self, recognizer_output): 
	if recognizer_output[0] == '-':
	    return self.previous_surname + "," + recognizer_output[2:]
	else:
	    iterator = recognizer_output.find(",")
	    self.previous_surname = recognizer_output[:iterator]
	    return recognizer_output
