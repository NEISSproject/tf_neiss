import tensorflow as tf
import tensorflow_datasets as tfds
import json


from input_fn.input_fn_nlp.input_fn_nlp_base import InputFnNLPBase
from input_fn.input_fn_nlp.util_input_fn_nlp.StringMapper import get_sm


class InputFnNER(InputFnNLPBase):
    def __init__(self, flags):
        self.add_types = [tf.int32 if type == "int" else tf.float32 for type in flags.add_types]
        super(InputFnNER, self).__init__(flags)
        print("start init input fct")
        self._tag_string_mapper = get_sm(self._flags.tags)

        self.get_shapes_types_defaults()

        self._tokenizer_de=tfds.features.text.SubwordTextEncoder.load_from_file(self._flags.tokenizer)
        self._tok_vocab_size=self._tokenizer_de.vocab_size



        print("done init input fct")


    def getTagMapper(self):
        return self._tag_string_mapper

    def get_num_tags(self):
        return self._tag_string_mapper.size()

    def get_shapes_types_defaults(self):

        input_shapes = {'sentence': [None]}

        tgt_shapes = {'tgt': [None],'targetmask':[None]}

        input_types = {'sentence': tf.int32}

        tgt_types = {'tgt': tf.int32,'targetmask':tf.int32}

        input_defaults = {'sentence': 0}

        tgt_defaults = {'tgt': 0,'targetmask':0}

        self._shapes = input_shapes, tgt_shapes
        self._types = input_types, tgt_types
        self._defaults = input_defaults, tgt_defaults

    def _parse_fn(self, training_data):
        sentence=''
        tags = []
        for j in range(len(training_data)):
            sentence=sentence + training_data[j][0]
            if j<len(training_data)-1:
                sentence=sentence + ' '
            tags.append(self._tag_string_mapper.get_channel(training_data[j][1]))
        enc_sentence=self._tokenizer_de.encode(sentence)
        tar_real=[]
        last_index=None
        curindex=len(enc_sentence)-1
        for j in range(len(training_data)-1,-1,-1):
            if last_index is not None:
                curlist=[last_index]
            else:
                curlist=[]
            while len(curlist)==0 or training_data[j][0] not in self._tokenizer_de.decode(curlist):
                curlist=[enc_sentence[curindex]]+curlist
                curindex-=1
            if last_index is not None:
                tar_real=(len(curlist)-1)*[tags[j]]+tar_real
                if self._flags.predict_mode:
                    training_data[j].append(len(curlist)-1)
            else:
                tar_real=len(curlist)*[tags[j]]+tar_real
                if self._flags.predict_mode:
                    training_data[j].append(len(curlist))

            last_subword=self._tokenizer_de.decode([curlist[0]])
            if len(last_subword)>2 and ' ' in last_subword[1:-1]:
                last_index=curlist[0]
            else:
                last_index=None
        #Add SOS-Tag and EOS-Tag
        enc_sentence=[self._tok_vocab_size]+enc_sentence+[self._tok_vocab_size+1]
        targetmask=[0]+len(tar_real)*[1] + [0]
        tar_real=[self.get_num_tags()]+tar_real+[self.get_num_tags()+1]
        if self._flags.predict_mode:
            inputs = {'sentence':[enc_sentence]}
            tgts = {'tgt': [tar_real],'targetmask':[targetmask]}
            return inputs, tgts, training_data
        inputs = {'sentence':enc_sentence}
        tgts = {'tgt': tar_real,'targetmask':targetmask}

        return inputs, tgts

    def generator_fn(self):
        for fname in self._fnames:
            with open(fname) as f:
                training_data=json.load(f)
                #print("Yield Sentence")
                for i in range(len(training_data)):
                    yield self._parse_fn(training_data[i])

    def generator_fn_predict(self,fname):
        with open(fname) as f:
            training_data=json.load(f)
            #print("Yield Sentence")
            for i in range(len(training_data)):
                yield self._parse_fn(training_data[i])



