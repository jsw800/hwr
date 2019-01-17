from postprocess_module import PostprocessModule
import editdistance as ed
from os.path import join

THIS_DIR_PATH = 'postprocessing/lastnamecorrection'

class LastNameCorrection(PostprocessModule):

    def __init__(self, args):
        dict_path = 'surname_dict.txt'
	with open(join(THIS_DIR_PATH, dict_path)) as f:
                self.dictionary = list(set([row for row in f.read().split('\n') if row != '']))

    def postprocess(self, recognizer_output):
	if hasattr(self, 'dictionary'): 
            if recognizer_output in self.dictionary:
                return recognizer_output
            min_ed = float('inf')
            best = None
            for label in self.dictionary:
                dist = ed.eval(label.lower(), recognizer_output.lower())
                if dist < min_ed:
                    min_ed = dist
                    best = label
	    #print("\n" + min_ed + " " + best + "\n")
	    if min_ed > (len(recognizer_output) / 2):
	        return recognizer_output
            return best
	else:
	    return recognizer_output
