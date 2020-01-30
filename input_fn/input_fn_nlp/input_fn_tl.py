import random

import tensorflow as tf
import tensorflow_datasets as tfds

from input_fn.input_fn_generator_base import InputFnBase


class InputFnTL(InputFnBase):
    """Input Function Generator for NLP problems, returns a dict..."""

    def __init__(self, flags):
        super(InputFnTL, self).__init__()
        self._flags = flags
        self._shapes = None
        self._types = None
        self._defaults = None
        self._shuffle = True
        self._max_length=40
        self._tokenizer_inp=tfds.features.text.SubwordTextEncoder.load_from_file(flags.tokenizer_inp)
        self._tokenizer_tar=tfds.features.text.SubwordTextEncoder.load_from_file(flags.tokenizer_tar)
        self.get_shapes_types_defaults()

    def encode(self,lang1, lang2):
        lang1 = [self._tokenizer_inp.vocab_size] + self._tokenizer_inp.encode(
            lang1.numpy()) + [self._tokenizer_inp.vocab_size+1]

        lang2 = [self._tokenizer_tar.vocab_size] + self._tokenizer_tar.encode(
            lang2.numpy()) + [self._tokenizer_tar.vocab_size+1]

        return lang1, lang2

    def tf_encode(self,pt, en):
        result_pt, result_en = tf.py_function(self.encode, [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])
        inputs={'inputs':result_pt,'tar_inp':result_en[:-1]}

        return inputs, result_en[1:]

    def filter_max_length(self,x, y):
        return tf.logical_and(tf.size(x['inputs']) <= self._max_length,
                        tf.size(y) <= self._max_length)

    def get_shapes_types_defaults(self):

        input_shapes = {'inputs':[None],'tar_inp':[None]}

        tgt_shapes = [None]

        input_types = {'inputs':tf.int64,'tar_inp':tf.int64}

        tgt_types = tf.int64

        input_defaults = {'inputs':tf.cast(0,tf.int64),'tar_inp':tf.cast(0,tf.int64)}

        tgt_defaults = tf.cast(0,tf.int64)

        self._shapes = input_shapes, tgt_shapes
        self._types = input_types, tgt_types
        self._defaults = input_defaults, tgt_defaults


    def generateDataSet(self, is_train_mode=True):

        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
        train_examples, val_examples = examples['train'], examples['validation']

        if is_train_mode:
            dataset = train_examples.map(self.tf_encode)
            dataset = dataset.filter(self.filter_max_length)
            # cache the dataset to memory to get a speedup while reading from it.
            dataset = dataset.cache()
            dataset = dataset.shuffle(self._flags.buffer).padded_batch(self._flags.train_batch_size, self._shapes, self._defaults)
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        else:
            dataset = val_examples.map(self.tf_encode)
            dataset = dataset.filter(self.filter_max_length).padded_batch(self._flags.val_batch_size, self._shapes, self._defaults)

        return dataset

    def get_input_vocab_size(self):
        return self._tokenizer_inp.vocab_size+2

    def get_target_vocab_size(self):
        return self._tokenizer_tar.vocab_size+2

    def get_input_fn_train(self):
        self._mode = 'train'
        return self.generateDataSet(True)

    def get_input_fn_val(self, image_preproc_thread_count=None):
        self._mode = 'val'
        return self.generateDataSet(False)

if __name__ == '__main__':
    tfds.features.text.SubwordTextEncoder.load_from_file("../../../data/tokenizer/tokenizer_pt")
