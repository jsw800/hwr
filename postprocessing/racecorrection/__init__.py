from postprocess_module import PostprocessModule
import editdistance as ed
from os.path import join

THIS_DIR_PATH = 'postprocessing/racecorrection'

class RaceCorrection(PostprocessModule):

    def __init__(self, args):
        pass

    def postprocess(self, recognizer_output):
        if recognizer_output[0] == 'W' or recognizer_output[0] == 'w':
            return "White"
	elif recognizer_output == "Neg" or recognizer_output == "neg":
	    return "Black"
	elif recognizer_output == "Mex" or recognizer_output == "mex":
	    return "Mexican"
	elif recognizer_output == "In" or recognizer_output == "in":
	    return "Indian"
	elif recognizer_output == "Ch" or recognizer_output == "ch":
            return "Chinese"
	elif recognizer_output == "Jp" or recognizer_output == "jp":
            return "Japanese"
	elif recognizer_output == "Fil" or recognizer_output == "fil":
            return "Filipino"
	elif recognizer_output == "Hin" or recognizer_output == "hin":
            return "Hindu"
	elif recognizer_output == "Kor" or recognizer_output == "kor":
            return "Korean"
        else:
            return recognizer_output

