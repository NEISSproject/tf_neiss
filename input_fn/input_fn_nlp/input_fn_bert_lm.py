from gensim.models import FastText

import tensorflow as tf
import tensorflow_datasets as tfds
import random


from input_fn.input_fn_nlp.input_fn_nlp_base import InputFnNLPBase
from input_fn.input_fn_nlp.util_input_fn_nlp.StringMapper import get_sm


class InputFnBertLM(InputFnNLPBase):
    def __init__(self, flags):
        self.add_types = [tf.int32 if type == "int" else tf.float32 for type in flags.add_types]
        super(InputFnBertLM, self).__init__(flags)
        print("start init input fct")
        self.get_shapes_types_defaults()

        #self._fasttextmodel = FastText.load_fasttext_format(self._flags.word_embeddings)

        self._tokenizer_de=tfds.features.text.SubwordTextEncoder.load_from_file(self._flags.tokenizer)
        self._tok_vocab_size=self._tokenizer_de.vocab_size
        self._max_token_text_part=self._flags.max_token_text_part



        print("done init input fct")

    def get_shapes_types_defaults(self):

        input_shapes = {'sentence': [None], 'sentencelength': [None],'masked_index': [None]}

        tgt_shapes = {'tgt': [None]}

        input_types = {'sentence': tf.int32, 'sentencelength': tf.int32,'masked_index': tf.int32}

        tgt_types = {'tgt': tf.int32}

        input_defaults = {'sentence': 0, 'sentencelength': 0,'masked_index': 0}

        tgt_defaults = {'tgt': 0}

        self._shapes = input_shapes, tgt_shapes
        self._types = input_types, tgt_types
        self._defaults = input_defaults, tgt_defaults

    def shorten_if_necessary(self,enc_list):
        listlen=len(enc_list)
        if listlen<=self._max_token_text_part:
            return enc_list
        splitindex=random.randint(0,listlen-self._max_token_text_part)
        shorter_list=enc_list[splitindex:splitindex+self._max_token_text_part]
        return shorter_list

    def _parse_fn(self, sentence):
        enc_sentence=self._tokenizer_de.encode(sentence)
        enc_sentence=self.shorten_if_necessary(enc_sentence)
        #Add SOS-Tag and EOS-Tag
        tar_real=[self._tok_vocab_size]+enc_sentence+[self._tok_vocab_size+1]
        masked_index_list=[0]
        word_index_list=[self._tok_vocab_size]
        #Masking
        for word_index in enc_sentence:
            masked_word_index, masked=self.mask_word_index(word_index)
            word_index_list.append(masked_word_index)
            masked_index_list.append(masked)
        word_index_list.append(self._tok_vocab_size+1)
        masked_index_list.append(0)


        #Here the sentencelength is only used in the loss function. Therefore it needs the length of the targets
        sentencelength=[len(tar_real)]
        inputs = {'sentence':word_index_list,'sentencelength':sentencelength,'masked_index':masked_index_list}
        tgts = {'tgt': tar_real}

        return inputs, tgts

    def generator_fn(self):
        for element in self._worklist:
            if len(element.split(' '))>10:
                yield self._parse_fn(element)

    def generator_fn_predict(self):
        for element in self._worklist:
            yield self._parse_fn(element)

    def mask_word_index(self,word_index):
        prob=random.random()
        if prob<=0.15:
            prob=prob/0.15
            if prob >0.2:
               #MASK-Token
               return self._tok_vocab_size+2,1
            elif prob>0.1:
               return random.randint(0,self._tok_vocab_size-1),1
            else:
                return word_index,1
        else:
            return word_index,0

if __name__ == '__main__':
    tokenizer_de=tfds.features.text.SubwordTextEncoder.load_from_file("../../../data/tokenizer/tokenizer_de")
    sample_string = 'Wie zum Henker funktioniert dieser Tokenizer?'

    tokenized_string = tokenizer_de.encode(sample_string)

    print('Tokenized string is {}'.format(tokenized_string))

    for ts in tokenized_string:
       print ('{} ----> {}'.format(ts, tokenizer_de.decode([ts])))


    original_string = tokenizer_de.decode(tokenized_string)
    print ('The original string: {}'.format(original_string))





