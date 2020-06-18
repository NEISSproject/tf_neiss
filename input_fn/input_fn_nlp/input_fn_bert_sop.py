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

        input_shapes = {'text': [None], 'masked_index': [None]}

        tgt_shapes = {'tgt_mlm': [None],'tgt_sop':[None]}

        input_types = {'text': tf.int32, 'masked_index': tf.int32}

        tgt_types = {'tgt_mlm': tf.int32,'tgt_sop':tf.int32}

        input_defaults = {'text': 0, 'masked_index': 0}

        tgt_defaults = {'tgt_mlm': 0,'tgt_sop':0}

        self._shapes = input_shapes, tgt_shapes
        self._types = input_types, tgt_types
        self._defaults = input_defaults, tgt_defaults

    def _parse_fn(self, sentences):
        first_enc_sentence=self._tokenizer_de.encode(sentences[0])
        sec_enc_sentence=self._tokenizer_de.encode(sentences[1])
        first_mask_enc_sentence,first_masked_index_list=self.mask_enc_sentence(first_enc_sentence)
        sec_mask_enc_sentence,sec_masked_index_list=self.mask_enc_sentence(sec_enc_sentence)
        #Add CLS-Tag and SEP-Tag
        if self.switch_sentences():
            text_index_list=[self._tok_vocab_size]+sec_mask_enc_sentence+[self._tok_vocab_size+1]+first_mask_enc_sentence+[self._tok_vocab_size+1]
            masked_index_list=[0]+sec_masked_index_list+[0]+first_masked_index_list+[0]
            tar_mlm=[self._tok_vocab_size]+sec_enc_sentence+[self._tok_vocab_size+1]+first_enc_sentence+[self._tok_vocab_size+1]
            tar_sop=[0]
        else:
            text_index_list=[self._tok_vocab_size]+first_mask_enc_sentence+[self._tok_vocab_size+1]+sec_mask_enc_sentence+[self._tok_vocab_size+1]
            masked_index_list=[0]+first_masked_index_list+[0]+sec_masked_index_list+[0]
            tar_mlm=[self._tok_vocab_size]+first_enc_sentence+[self._tok_vocab_size+1]+sec_enc_sentence+[self._tok_vocab_size+1]
            tar_sop=[1]
        inputs = {'text':text_index_list,'masked_index':masked_index_list}
        tgts = {'tgt_mlm': tar_mlm,'tgt_sop': tar_sop}

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

    def mask_enc_sentence(self,enc_sentence):
        masked_index_list=[]
        word_index_list=[]
        #Masking
        for word_index in enc_sentence:
            masked_word_index, masked=self.mask_word_index(word_index)
            word_index_list.append(masked_word_index)
            masked_index_list.append(masked)
        return word_index_list,masked_index_list

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
    sentencelength=[1,2,3,4,5]
    weights = tf.sequence_mask(sentencelength)
    print(weights)





