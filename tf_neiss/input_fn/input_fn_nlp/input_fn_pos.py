from gensim.models import FastText

import tensorflow as tf


from input_fn.input_fn_nlp.input_fn_nlp_base import InputFnNLPBase
from input_fn.input_fn_nlp.util_input_fn_nlp.StringMapper import get_sm


class InputFnPOS(InputFnNLPBase):
    def __init__(self, flags):
        self.add_types = [tf.int32 if type == "int" else tf.float32 for type in flags.add_types]
        super(InputFnPOS, self).__init__(flags)
        print("start init input fct")
        self._tag_string_mapper = get_sm(self._flags.tags)

        self.get_shapes_types_defaults()

        self._fasttextmodel = FastText.load_fasttext_format(self._flags.word_embeddings)


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

        input_shapes = {'sentence': [None,None], 'sentencelength': [None]}

        tgt_shapes = {'tgt': [None]}

        input_types = {'sentence': tf.float32, 'sentencelength': tf.int32}

        tgt_types = {'tgt': tf.int32}

        input_defaults = {'sentence': 0.0, 'sentencelength': 0}

        tgt_defaults = {'tgt': 0}

        #for idx, type in enumerate(self.add_types):
        #    input_types['feat{}'.format(idx)] = type
        #    input_shapes['feat{}'.format(idx)] = [None]
        #    input_defaults['feat{}'.format(idx)] = 0 if type == tf.int32 else 0.0

        self._shapes = input_shapes, tgt_shapes
        self._types = input_types, tgt_types
        self._defaults = input_defaults, tgt_defaults

    def _parse_fn(self, inputlist, targetlist):

        sentence=[]
        tags = []
        for j in range(len(inputlist)):
            sentence.append(self._fasttextmodel[inputlist[j]])
            if targetlist is not None:
                tags.append(self._tag_string_mapper.get_channel(targetlist[j]))
        sentencelength=[len(sentence)]

        #print("Shape sentence", len(sentence),len(sentence[0]),sentence[0])

        inputs = {'sentence':sentence,'sentencelength':sentencelength}
        if targetlist is not None:
            assert len(sentence) == len(tags), "Words and tags lengths don't match"
        tgts = {'tgt': tags}

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


