import tensorflow as tf
import tensorflow_datasets as tfds
import random
import json


from input_fn.input_fn_nlp.input_fn_nlp_base import InputFnNLPBase


class InputFnBertNSP(InputFnNLPBase):
    def __init__(self, flags):
        self.add_types = [tf.int32 if type == "int" else tf.float32 for type in flags.add_types]
        super(InputFnBertNSP, self).__init__(flags)
        print("start init input fct")
        self.get_shapes_types_defaults()

        self._tokenizer_de=tfds.features.text.SubwordTextEncoder.load_from_file(self._flags.tokenizer)
        self._tok_vocab_size=self._tokenizer_de.vocab_size



        print("done init input fct")

    def get_shapes_types_defaults(self):

        input_shapes = {'text': [None], 'masked_index': [None]}

        tgt_shapes = {'tgt_mlm': [None],'tgt_nsp':[None]}

        input_types = {'text': tf.int32, 'masked_index': tf.int32}

        tgt_types = {'tgt_mlm': tf.int32,'tgt_nsp':tf.int32}

        input_defaults = {'text': 0, 'masked_index': 0}

        tgt_defaults = {'tgt_mlm': 0,'tgt_nsp':0}

        self._shapes = input_shapes, tgt_shapes
        self._types = input_types, tgt_types
        self._defaults = input_defaults, tgt_defaults

    def _parse_fn(self, sentences):
        take_connected_parts=self.bool_decision()
        firstinputlist=sentences[0].split(' ')
        nofirstwords=len(firstinputlist)
        #minimal word number is 10
        splitindex=random.randint(4,nofirstwords-5)
        textpartone=firstinputlist[:splitindex]
        #maximal text sequence length is 40
        if len(textpartone)>40:
            textpartone=textpartone[len(textpartone)-40:]
        if take_connected_parts:
            textparttwo=firstinputlist[splitindex:]
            tar_nsp=[1]
        else:
            secondinputlist=sentences[1].split(' ')
            nosecondwords=len(secondinputlist)
            splitindex=random.randint(0,nosecondwords-5)
            textparttwo=secondinputlist[splitindex:]
            tar_nsp=[0]
        if len(textparttwo)>40:
            textparttwo=textparttwo[:40]
        textpartone=' '.join(textpartone)
        textparttwo=' '.join(textparttwo)
        first_enc_sentence=self._tokenizer_de.encode(textpartone)
        sec_enc_sentence=self._tokenizer_de.encode(textparttwo)
        first_mask_enc_sentence,first_masked_index_list=self.mask_enc_sentence(first_enc_sentence)
        sec_mask_enc_sentence,sec_masked_index_list=self.mask_enc_sentence(sec_enc_sentence)
        switch_order=self.bool_decision()
        #Add CLS-Tag and SEP-Tag
        if switch_order:
            text_index_list=[self._tok_vocab_size]+sec_mask_enc_sentence+[self._tok_vocab_size+1]+first_mask_enc_sentence+[self._tok_vocab_size+1]
            masked_index_list=[0]+sec_masked_index_list+[0]+first_masked_index_list+[0]
            tar_mlm=[self._tok_vocab_size]+sec_enc_sentence+[self._tok_vocab_size+1]+first_enc_sentence+[self._tok_vocab_size+1]
        else:
            text_index_list=[self._tok_vocab_size]+first_mask_enc_sentence+[self._tok_vocab_size+1]+sec_mask_enc_sentence+[self._tok_vocab_size+1]
            masked_index_list=[0]+first_masked_index_list+[0]+sec_masked_index_list+[0]
            tar_mlm=[self._tok_vocab_size]+first_enc_sentence+[self._tok_vocab_size+1]+sec_enc_sentence+[self._tok_vocab_size+1]
        inputs = {'text':text_index_list,'masked_index':masked_index_list}
        tgts = {'tgt_mlm': tar_mlm,'tgt_nsp': tar_nsp}

        return inputs, tgts

    def generator_fn(self):
        for fname in self._fnames:
            with open(fname, 'r',encoding="utf-8") as f:
                nsplist = json.load(f)
                for element in nsplist:
                    yield self._parse_fn(element)

    def generator_fn_predict(self):
        for fname in self._fnames:
            with open(fname, 'r',encoding="utf-8") as f:
                nsplist = json.load(f)
                for element in nsplist:
                    yield self._parse_fn(element)

    def bool_decision(self):
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





