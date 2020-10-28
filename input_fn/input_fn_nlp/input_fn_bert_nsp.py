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
        self._max_words_text_part=self._flags.max_words_text_part
        self._max_token_text_part=self._flags.max_token_text_part



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

    def build_two_sentence_segments(self,articles,take_connected_parts):
        if take_connected_parts:
            sentences=articles[0]
            lensentences=len(sentences)
            splitindex=random.randint(0,lensentences-2)
            first_sentences=sentences[splitindex]
            second_sentences=sentences[splitindex+1]
        else:
            first_article=articles[0]
            second_article=articles[1]
            splitindex=random.randint(0,len(first_article)-2)
            first_sentences=first_article[splitindex]
            splitindex2=random.randint(0,len(second_article)-2)
            second_sentences=second_article[splitindex2+1]
            sentences=first_article[:splitindex+1]+second_article[splitindex2+1:]
            lensentences=len(sentences)
        first_enc_sentence=self._tokenizer_de.encode(first_sentences)
        second_enc_sentence=self._tokenizer_de.encode(second_sentences)
        firstaddindex=splitindex-1
        secondaddindex=splitindex+2

        #Check if it is already to long
        if len(first_enc_sentence)+len(second_enc_sentence)>self._max_token_text_part:
            half=int(self._max_token_text_part/2)
            if len(first_enc_sentence)>half:
                first_enc_sentence=first_enc_sentence[len(first_enc_sentence)-half:]
            if len(second_enc_sentence)>half:
                second_enc_sentence=second_enc_sentence[:half]
        else:
            #Attempt to extend
            stop=False
            while not stop:
                if firstaddindex<0 and secondaddindex>=lensentences:
                    stop=True
                elif firstaddindex<0:
                    stopback=False
                    while not stopback:
                        new_sentences=second_sentences+' '+sentences[secondaddindex]
                        new_enc_sentence=self._tokenizer_de.encode(new_sentences)
                        if len(first_enc_sentence)+len(new_enc_sentence)<=self._max_token_text_part:
                            second_sentences=new_sentences
                            second_enc_sentence=new_enc_sentence
                            secondaddindex+=1
                            if secondaddindex>=lensentences:
                                stopback=True
                        else:
                            stopback=True
                    stop=True
                elif secondaddindex>=lensentences:
                    stopfront=False
                    while not stopfront:
                        new_sentences=sentences[firstaddindex]+' '+first_sentences
                        new_enc_sentence=self._tokenizer_de.encode(new_sentences)
                        if len(second_enc_sentence)+len(new_enc_sentence)<=self._max_token_text_part:
                            first_sentences=new_sentences
                            first_enc_sentence=new_enc_sentence
                            firstaddindex-=1
                            if firstaddindex<0:
                                stopfront=True
                        else:
                            stopfront=True
                    stop=True
                else:
                    if random.choice([True, False]):
                        new_sentences=sentences[firstaddindex]+' '+first_sentences
                        new_enc_sentence=self._tokenizer_de.encode(new_sentences)
                        if len(first_enc_sentence)+len(second_enc_sentence)+len(new_enc_sentence)<=self._max_token_text_part:
                            first_sentences=new_sentences
                            first_enc_sentence=new_enc_sentence
                            firstaddindex-=1
                        else:
                            firstaddindex=-1
                    else:
                        new_sentences=second_sentences+' '+sentences[secondaddindex]
                        new_enc_sentence=self._tokenizer_de.encode(new_sentences)
                        if len(first_enc_sentence)+len(second_enc_sentence)+len(new_enc_sentence)<=self._max_token_text_part:
                            second_sentences=new_sentences
                            second_enc_sentence=new_enc_sentence
                            secondaddindex+=1
                        else:
                            secondaddindex=lensentences
        return first_enc_sentence,second_enc_sentence

    def _parse_fn(self, sentences):
        take_connected_parts=self.bool_decision()
        if self._flags.segment_train:
            firstinputlist=sentences[0].split(' ')
            nofirstwords=len(firstinputlist)
            #minimal word number is 10
            splitindex=random.randint(4,nofirstwords-5)
            textpartone=firstinputlist[:splitindex]
            #maximal text sequence length is 40
            if len(textpartone)>self._max_words_text_part:
                textpartone=textpartone[len(textpartone)-self._max_words_text_part:]
            if take_connected_parts:
                textparttwo=firstinputlist[splitindex:]
                tar_nsp=[1]
            else:
                secondinputlist=sentences[1].split(' ')
                nosecondwords=len(secondinputlist)
                splitindex=random.randint(0,nosecondwords-5)
                textparttwo=secondinputlist[splitindex:]
                tar_nsp=[0]
            if len(textparttwo)>self._max_words_text_part:
                textparttwo=textparttwo[:self._max_words_text_part]
            textpartone=' '.join(textpartone)
            textparttwo=' '.join(textparttwo)
            first_enc_sentence=self._tokenizer_de.encode(textpartone)
            sec_enc_sentence=self._tokenizer_de.encode(textparttwo)
            #Check if it is too long
            if len(first_enc_sentence)+len(sec_enc_sentence)>self._max_token_text_part:
                half=int(self._max_token_text_part/2)
                if len(first_enc_sentence)>half:
                    first_enc_sentence=first_enc_sentence[len(first_enc_sentence)-half:]
                if len(sec_enc_sentence)>half:
                    sec_enc_sentence=sec_enc_sentence[:half]
        else:
            first_enc_sentence, sec_enc_sentence=self.build_two_sentence_segments(sentences,take_connected_parts)
            if take_connected_parts:
                tar_nsp=[1]
            else:
                tar_nsp=[0]
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
        for element in self._worklist:
            yield self._parse_fn(element)

    def generator_fn_predict(self):
        for element in self._worklist:
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





