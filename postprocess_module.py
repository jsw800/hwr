from abc import ABC, abstractmethod

"""
This is the abstract base class for a postprocess module.
Every postprocess module should override the methods here, or it will
not work. This serves more as a documentation file than anything.
"""


class PostprocessModule(ABC):

    @abstractmethod
    def __init__(self, args):
        pass

    @abstractmethod
    def postprocess(self, recognizer_output):
        pass
