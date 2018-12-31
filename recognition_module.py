from abc import ABC, abstractmethod

"""
This is the abstract base class for a recognition module.
Every recognition module should override the methods here,
or it will not work. This serves more as a documentation
file than anything.

The recognition module is responsible for all recognition work,
it just receives the raw image as it was cropped by the segmenter
and is responsible for recognizing it.
"""


class RecognitionModule(ABC):

    @abstractmethod
    def __init__(self, args):
        pass

    # perform training
    @abstractmethod
    def train(self):
        pass

    # Check we can use the module (e.g. is there a weights file?)
    @abstractmethod
    def is_trained(self):
        pass

    # produces output for a **single** image
    @abstractmethod
    def run(self, input_image):
        pass

