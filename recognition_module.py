from abc import ABCMeta, abstractmethod

"""
This is the abstract base class for a recognition module.
Every recognition module should override the methods here,
or it will not work. This serves more as a documentation
file than anything.

The recognition module is responsible for all recognition work,
it just receives the raw image as it was cropped by the segmenter
and is responsible for recognizing it.
"""


class RecognitionModule:
    __metaclass__ = ABCMeta

    """
    Each recognition module can have whatever args it wants, those are just passed in
    as a tuple. Document what args are expected in the concrete recognition module classes
    """
    @abstractmethod
    def __init__(self, args):
        pass

    # produces output for a **single** image
    @abstractmethod
    def run(self, input_image):
        pass

