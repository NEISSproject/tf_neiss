from gensim.models import FastText

import tensorflow as tf
import tensorflow_datasets as tfds
import random
import json


from input_fn.input_fn_nlp.input_fn_nlp_base import InputFnNLPBase


class InputFnBertSOP(InputFnNLPBase):
    def __init__(self, flags):
        self.add_types = [tf.int32 if type == "int" else tf.float32 for type in flags.add_types]
        super(InputFnBertSOP, self).__init__(flags)
        print("start init input fct")
        self.get_shapes_types_defaults()

        self._tokenizer_de=tfds.features.text.SubwordTextEncoder.load_from_file(self._flags.tokenizer)
        self._tok_vocab_size=self._tokenizer_de.vocab_size



        print("done init input fct")

    def get_shapes_types_defaults(self):

        input_shapes = {'text': [None], 'textlength': [None]}

        tgt_shapes = {'tgt': [None]}

        input_types = {'text': tf.int32, 'textlength': tf.int32}

        tgt_types = {'tgt': tf.int32}

        input_defaults = {'text': 0, 'textlength': 0}

        tgt_defaults = {'tgt': 0}

        self._shapes = input_shapes, tgt_shapes
        self._types = input_types, tgt_types
        self._defaults = input_defaults, tgt_defaults

    def _parse_fn(self, sentences):
        first_enc_sentence=self._tokenizer_de.encode(sentences[0])
        sec_enc_sentence=self._tokenizer_de.encode(sentences[1])
        #Add CLS-Tag and SEP-Tag
        if self.switch_sentences():
            text_index_list=[self._tok_vocab_size]+sec_enc_sentence+[self._tok_vocab_size+1]+first_enc_sentence
            tar=[0]*len(text_index_list)
        else:
            text_index_list=[self._tok_vocab_size]+first_enc_sentence+[self._tok_vocab_size+1]+sec_enc_sentence
            tar=[1]*len(text_index_list)
        textlength=[len(tar)]


        inputs = {'text':text_index_list,'textlength':textlength}
        tgts = {'tgt': tar}

        return inputs, tgts

    def generator_fn(self):
        for fname in self._fnames:
            with open(fname, 'r',encoding="utf-8") as f:
                soplist = json.load(f)
                for element in soplist:
                    yield self._parse_fn(element)

    def generator_fn_predict(self):
        for fname in self._fnames:
            with open(fname, 'r',encoding="utf-8") as f:
                soplist = json.load(f)
                for element in soplist:
                    yield self._parse_fn(element)

    def switch_sentences(self):
        return random.choice([True, False])

if __name__ == '__main__':
    sentencelength=[1,2,3,4,5]
    weights = tf.sequence_mask(sentencelength)
    print(weights)





