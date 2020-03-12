from gensim.models import FastText

import tensorflow as tf
import tensorflow_datasets as tfds


from input_fn.input_fn_nlp.input_fn_nlp_base import InputFnNLPBase
from input_fn.input_fn_nlp.util_input_fn_nlp.StringMapper import get_sm


class InputFnTFPOS(InputFnNLPBase):
    def __init__(self, flags):
        self.add_types = [tf.int32 if type == "int" else tf.float32 for type in flags.add_types]
        super(InputFnTFPOS, self).__init__(flags)
        print("start init input fct")
        self._tag_string_mapper = get_sm(self._flags.tags)

        self.get_shapes_types_defaults()

        #self._fasttextmodel = FastText.load_fasttext_format(self._flags.word_embeddings)

        self._tokenizer_de=tfds.features.text.SubwordTextEncoder.load_from_file(self._flags.tokenizer)
        self._tok_vocab_size=self._tokenizer_de.vocab_size



        print("done init input fct")


    def get_num_words(self):
        return self._word_string_mapper.size()


    # def get_num_bigrams(self):
    #     return self._bigram_string_mapper.size()
    #
    # def get_num_trigrams(self):
    #     return self._trigram_string_mapper.size()
    def getTagMapper(self):
        return self._tag_string_mapper

    def get_num_tags(self):
        return self._tag_string_mapper.size()

    def get_shapes_types_defaults(self):

        input_shapes = {'sentence': [None], 'sentencelength': [None],'tar_inp':[None]}

        tgt_shapes = {'tgt': [None]}

        input_types = {'sentence': tf.int32, 'sentencelength': tf.int32,'tar_inp':tf.int32}

        tgt_types = {'tgt': tf.int32}

        input_defaults = {'sentence': 0, 'sentencelength': 0,'tar_inp':0}

        tgt_defaults = {'tgt': 0}

        #for idx, type in enumerate(self.add_types):
        #    input_types['feat{}'.format(idx)] = type
        #    input_shapes['feat{}'.format(idx)] = [None]
        #    input_defaults['feat{}'.format(idx)] = 0 if type == tf.int32 else 0.0

        self._shapes = input_shapes, tgt_shapes
        self._types = input_types, tgt_types
        self._defaults = input_defaults, tgt_defaults

    def _parse_fn(self, inputlist, targetlist):

        sentence=''
        tags = []
        for j in range(len(inputlist)):
            sentence=sentence + inputlist[j]
            if j<len(inputlist)-1:
                sentence=sentence + ' '
            if targetlist is not None:
                tags.append(self._tag_string_mapper.get_channel(targetlist[j]))
        enc_sentence=self._tokenizer_de.encode(sentence)
        if "BERT" in self._flags.graph:
            tar_real=[]
            last_index=None
            curindex=len(enc_sentence)-1
            for j in range(len(inputlist)-1,-1,-1):
                if last_index is not None:
                    curlist=[last_index]
                else:
                    curlist=[]
                while len(curlist)==0 or inputlist[j] not in self._tokenizer_de.decode(curlist):
                    curlist=[enc_sentence[curindex]]+curlist
                    curindex-=1
                if last_index is not None:
                    tar_real=(len(curlist)-1)*[tags[j]]+tar_real
                else:
                    tar_real=len(curlist)*[tags[j]]+tar_real

                last_subword=self._tokenizer_de.decode([curlist[0]])
                if len(last_subword)>2 and ' ' in last_subword[1:-1]:
                    last_index=curlist[0]
                else:
                    last_index=None
        #Add SOS-Tag and EOS-Tag
        enc_sentence=[self._tok_vocab_size]+enc_sentence+[self._tok_vocab_size+1]


        #Add SOS-Tag for input targets
        tar_inp=[self.get_num_tags()]+tags

        if "BERT" in self._flags.graph:
            tar_real=[self.get_num_tags()]+tar_real+[self.get_num_tags()+1]
        else:
            #Add EOS-Tag for real targets
            tar_real=tags+[self.get_num_tags()+1]

        #In Transformer the sentencelength is only used in the loss function. Therefore it needs the length of the targets
        sentencelength=[len(tar_real)]
        inputs = {'sentence':enc_sentence,'sentencelength':sentencelength,'tar_inp':tar_inp}
        #if targetlist is not None:
        #    assert len(sentence) == len(tags), "Words and tags lengths don't match"
        tgts = {'tgt': tar_real}

        return inputs, tgts

    def generator_fn(self):

        if self._flags.predict_mode:
            for fname in self._fnames:
                with open(fname, 'r',encoding="utf-8") as f:
                    setlist=f.readlines()
                    setlist=[setlist[i].replace('\n','').split(sep=" ") for i in range(len(setlist))]
                    inputlist=[[setlist[i][j] for j in range(len(setlist[i]))] for i in range(len(setlist))]
                    for i in range(len(inputlist)):
                        yield self._parse_fn(inputlist[i],None)
        else:
            for fname in self._fnames:
                with open(fname, 'r',encoding="utf-8") as f:
                    setlist=f.readlines()
                    setlist=[setlist[i].replace('\n','').split(sep=" ") for i in range(len(setlist))]
                    inputlist=[[setlist[i][j].split(sep="<#>")[0] for j in range(len(setlist[i]))] for i in range(len(setlist))]
                    targetlist=[[setlist[i][j].split(sep="<#>")[1] for j in range(len(setlist[i]))] for i in range(len(setlist))]
                    for i in range(len(inputlist)):
                        #print("Yield Sentence")
                        yield self._parse_fn(inputlist[i],targetlist[i])

    def generator_fn_predict(self):

        for fname in self._fnames:
            with open(fname, 'r',encoding="utf-8") as f:
                setlist=f.readlines()
                setlist=[setlist[i].replace('\n','').split(sep=" ") for i in range(len(setlist))]
                inputlist=[[setlist[i][j] for j in range(len(setlist[i]))] for i in range(len(setlist))]
                for i in range(len(inputlist)):
                    yield self._parse_fn(inputlist[i],None)


