# general
* use flake8 and black. Lots of unused imports etc
* Unnecessary use of __dict__[] e.g. in util/flags FlagValues:
    * Circumventing __setattr__ and __getattr__
    * Much more complicated than necessary
* avoid global variables. Much overused
* move from script collection to api. This is crucial e.g. for metalearning
* have a look at dvi pipelines. It solves a lot of what is beeing done here with full reproducibility
* Commenting out large blocks of code in major commits is confusing and trashes the repo e.g. graphs_2d.py
* I removed everything which seems very specific to certain projects to me. In case I threw out very general stuff it should be integrated again but within more generic naming hierarchy

# level problematic for packaging
* multiple modules (util, trainer ...) on highest hierarchy
* should be all part of a subfolder tf_neiss

# test/test_requirements.py
* is not a test but an installation procedure. I added an example of a proper unittest under tests/test_package.py
* not necessary -> setup.cfg install_requires
* now: pip install -e .
* Missing tensorflow and tensorflow_addons requirements (added)
* TODO: look through install_requires and through out what is not required actually. Due to the stuffing of this package with loads of special cases this ling could possibly be much longer than required.

# test/workdir_triangle2d/**/README.md
* In case these are just placeholders for git to generate the folder, use a file called .gitkeep which is the convention

# test/workdir_triangle2d/test_triangle2d_full.sh
* Is not agnostic to the location where it is run (especially not from base dir. Requires ```cd tests/workdir_triangle2d``` to run).

# test/workdir_triangle2d/test_triangle2d_gpu_full.sh
* see above

# util/misc
* not helpful as a name. Should be called log or logging because that's what it mostly contains.
* tee: is not necessary. The Tee class is what is called a handler in logging.
    * Use logging.info / logging.debug ... instead of print statements
* get_commit_id: Good idea to log the commit_id but this becomes unnecessary when using dvc

# util/flags
* Avoid global variables
    * Global FLAGS dangerous, especially when overwriting on reload in e.g. trainer/trainer_base
        * Can lead to horrible to debug bugs
        * I even had a bug related to this when trying to get the  triangle2d example running: missing flags.FLAGS.parse_flags() at the end of the flags section in generate_train_data_2d_triangle_TFR.py
        * rather pass the unparsed parser around and parse it finally in the train() method
    * get rid of global_parser
        * make all the helper methods
* add_argument wrappers: Rather complicated wrapping without much benefit
    * In any case it should not modify a global parser but modify a parser passed to the function.
    * Could well be methods of your class inheriting argparse.ArgumentParser
    * Note: Could be produced in method/function factories from map
* Predefined options is a good idea but this should be enforced by kwargs parsing in trainer/trainer_base TrainerBase.__init__(self, **kwargs) or equivalent base classes
    * allows flexibility to use tf_neiss as api and not only as script
    * Add a method to each trainer that returns an ArgumentParser
* module level definition of parser options is a dangerous idea. I head to comment out three flag definitions in lav_base.py because if you import multiple files at the same time everything blows up (val_list and further were defined twice then. Finding where these are defined remains a TODO and is a good example why it is a bad idea to use global variables for that.)

# util/io_helpers + utils/list_helpers
* looks like overabstraction to me concerning the limited use of e.g. IOContext
* I removed the io_helpers and list_helpers files for now. TODO: You can tell me where these files should go and I will put them in again.

# util/tools/split_train_val
* This is a binary, not a library helpers collection like everything else in util
* Highly specific and hardcoded (e.g. 500)

# model_fn/util_model_fn
* this is probably the most useful collection for the NEISS community
* moved to tf_neiss/models
* confusing content e.g.
    * custom_layers.py contains one keras Layer which one would expect to be in keras_compatible_layers.py
    * keras_compatible_layers.py contains contains also simple tf activation functions which one would expect to be in losses.py
        * Also these definitions are completely overdone.
            e.g.
            ```
            def relu(features, name=None):
                return tf.keras.activations.relu(features, name=name)
            ```
            is equivalent to 
            ```
            from tfk.keras.activations import relu
            ```
    * I renamed custom_layers.py to layers.py
    * renamed misc.py to transformation.py
    * renamed optimizer.py to optimizers.py (consistency with neighbour files)
    * It remains a TODO to split the content of keras_compatible_layers.py into layers.py, losses.py, transform.py, optimizers.py and maybe further files like activation

# input_fn/
* renamed input_fn_gerator_base.py to input_fn_base.py for consistency

# module namespace
* Added the most basic classes to module level namespace (see tf_neiss/__init__.py)

# example with triangle2d
* I moved one set of specifiy to examples/triangles_2d
* I tried to make the bash script work but tensorflow tells me, the model was not built. TODO: This needs to be fixed.
* This example could become a proper test by adding all stages as separate tests (real unittests) under tests/

# Merging *base.py
* I would further propose to merge all the definitions in *_base.py files into one file e.g. bases.py or core.py
