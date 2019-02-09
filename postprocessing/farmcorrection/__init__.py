from postprocess_module import PostprocessModule
import editdistance as ed
from os.path import join

THIS_DIR_PATH = 'postprocessing/farm'

class FarmCorrection(PostprocessModule):

    def __init__(self, args):
        pass

    def postprocess(self, recognizer_output): 
	if recognizer_output[0] == 'V' or recognizer_output[0] == 'v' or recognizer_output[0] == 'X' or recognizer_output[0] == 'x':
	    return self.previous_output
	elif recognizer_output == "Yes" or recognizer_output == "No":
	    self.previous_output = recognizer_output
	    return recognizer_output
	elif len(recognizer_output) == 0 or recognizer_output == "--":
	    return "blank"
	else:
	    return "invalid input"
	   
