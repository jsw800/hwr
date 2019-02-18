from abc import ABCMeta, abstractmethod

"""
This is the abstract base class for a preprocess module.
Every postprocess module should override the methods here, or it will
not work. This serves more as a documentation file than anything.
"""


class PreprocessModule:
    __metaclass__ = ABCMeta

    """
    Each postprocess module can define what args it wants. They are passed
    in as a single tuple. Document expected args inside each postprocess module.
    """
    @abstractmethod
    def __init__(self, args):
        pass

    @abstractmethod
    def preprocess(self, input_image):
        pass

    @abstractmethod
    def batch_preprocess(self, input_image):
        pass
