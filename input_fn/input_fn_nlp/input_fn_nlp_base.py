import random

import tensorflow as tf

from input_fn.input_fn_generator_base import InputFnBase


class InputFnNLPBase(InputFnBase):
    """Input Function Generator for NLP problems, returns a dict..."""

    def __init__(self, flags):
        super(InputFnNLPBase, self).__init__()
        self._flags = flags
        self._shapes = None
        self._types = None
        self._defaults = None
        self._shuffle = True
        self._fnames = []

    def get_shapes_types_defaults(self):
        pass

    def generator_fn(self):
        pass

    def generateDataSet(self, is_train_mode=True):

        if is_train_mode:
            dataset = tf.data.Dataset.from_generator(self.generator_fn,
                                                     output_shapes=self._shapes, output_types=self._types)
            if self._shuffle:
                dataset = dataset.shuffle(self._flags.buffer)
            dataset = dataset.repeat()
            dataset = (dataset
                       .padded_batch(self._flags.train_batch_size, self._shapes, self._defaults)
                       .prefetch(1)
                       )
        else:
            dataset = tf.data.Dataset.from_generator(self.generator_fn,
                                                     output_shapes=self._shapes, output_types=self._types)
            dataset = (dataset
                       .padded_batch(self._flags.val_batch_size, self._shapes, self._defaults)
                       .prefetch(1))

        return dataset

    def get_input_fn_train(self):
        self._mode = 'train'
        self._fnames = []
        for flist in self._flags.train_lists:
            with open(flist, 'r') as f:
                # print(f.read().splitlines())
                self._fnames.extend(f.read().splitlines())
        random.shuffle(self._fnames)
        return self.generateDataSet(True)

    def get_input_fn_val(self, image_preproc_thread_count=None):
        self._mode = 'val'
        self._fnames = []
        # print("start val")

        with open(self._flags.val_list, 'r') as f:
            self._fnames.extend(f.read().splitlines())
        return self.generateDataSet(False)

    def get_input_fn_predict(self):
        self._mode = 'predict'
        self._fnames = []
        with open(self._flags.predict_list, 'r') as f:
            self._fnames.extend(f.read().splitlines())
        return self.generateDataSet(False)
