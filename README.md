# hwr
Generalized handwriting recognizer for census documents (eventually).

`run.py`:
Given a yaml config path, image root directory, segmentation path for that image dir, and output filename,
this script will load the specified recognition and postprocessing modules for each field in the yaml config file.
It will then use the provided segmentation data to read the images and output the labels to the specified output file.

`load_modules.py`:
Responsible for loading recognition and postprocessing modules given the yaml config path.
Also, when a new recognition module or postprocessing module is implemented, it should be
imported into this file and registered in the `RECOGNITION_MODULE_NAMES` or `POSTPROCESS_MODULE_NAMES`
dictionaries.

`recognition_module.py`:
Defines the interface a recognition module must implement. Every recognition module
should inherit this class and implement the methods in it - `__init__(self, args)` and
`run(self, input_image)`

`postprocess_module.py`:
Defines the interface a postprocess module must implement. Every postprocess module should inherit
this class and implement the methods in it - `__init__(self, args)` and `run(self, recognizer_output)`

`printer.py`:
Defines a csv printer that is used to output the generated labelled data to a .csv file. Feel free
to define a different class with the same method interface if you want to redefine how output is being handled.

`datasets/Dataset.py`:
Given segmentation data and the image root directory, this class loads and crops census images **line by line**.
`__getitem__` ([] operator) returns a dictionary with the name of the image  the line is in (`image_name`), along
with a list of images, one for each field defined in the segmentation data.



# Dependencies
warp-ctc: https://github.com/SeanNaren/warp-ctc

This is a dependency that many of the recognition modules may end up using. We'll include it globally so any new
recognition modules that use CTC won't need their own copy.


Recognition module classes should be stored in `recognition/[module_name_here]` directory
(please put them in their own directory)

Postprocess module classes should be stored in `postprocessing/[module_name_here]` directory
(please put them in their own directory)


`configs` directory: put census configs here. These are yaml files defining which recognition and
postprocessing modules will be used for each individual field. Use `configs/none_config.yaml` as an example.


When you want to create a new recognition module, all you *need* to include is the `RecognitionModule` class that
defines how it will read images it is given. However, you may, and probably will want to include all the training
scripts for it inside the directory for that module, even though those training scripts will never be called directly
by this system, but rather by you as you prepare the networks your modules will use. As long as you provide a working
`RecognitionModule` class to be loaded by `load_modules.py`, you have free reign to include any and all training and
supporting scripts in the directory for your recognition module. *DON'T FORGET TO REGISTER ANY MODULES YOU WRITE IN*
`load_modules.py`.

Feel free to use the `recognition_args` attribute of each field to pass in a path to a network config file, and/or
any other info you may want to pass to your recognition module. These args will be passed to your recognition module
class as a tuple. Just be sure to document what args are expected in your module file.


Postprocessing modules are just like recognition modules, although they are less likely to need training scripts,
and they take strings as input rather than images. They also receive arguments as a tuple, and you can decide how
and if you need to use those arguments for anything. You may also feel free to include any and all supporting/training
scripts in the directories for your postprocessing module, if you need to. *DON'T FORGET TO REGISTER ANY MODULES YOU WRITE
IN* `load_modules.py`

