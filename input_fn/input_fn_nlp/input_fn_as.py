import tensorflow as tf
import tensorflow_datasets as tfds
import random
import json


from input_fn.input_fn_nlp.input_fn_nlp_base import InputFnNLPBase


class InputFnAS(InputFnNLPBase):
    def __init__(self, flags):
        self.add_types = [tf.int32 if type == "int" else tf.float32 for type in flags.add_types]
        super(InputFnAS, self).__init__(flags)
        print("start init input fct")
        self.get_shapes_types_defaults()
        self.get_shapes_types_defaults_predict()

        self._tokenizer_de=tfds.features.text.SubwordTextEncoder.load_from_file(self._flags.tokenizer)
        self._tok_vocab_size=self._tokenizer_de.vocab_size
        self._max_token_length=self._flags.max_tokens_text_part



        print("done init input fct")

    def get_shapes_types_defaults(self):

        input_shapes = {'text': [None]}

        tgt_shapes = {'tgt_as':[None]}

        input_types = {'text': tf.int32}

        tgt_types = {'tgt_as':tf.int32}

        input_defaults = {'text': 0}

        tgt_defaults = {'tgt_as':0}

        self._shapes = input_shapes, tgt_shapes
        self._types = input_types, tgt_types
        self._defaults = input_defaults, tgt_defaults

    def get_shapes_types_defaults_predict(self):

        input_shapes = {'text': [None]}

        tgt_shapes = {'tgt_as':[None]}

        el_shapes = {'pid':[None],'tb_id0':[None],'tb_id1':[None]}

        input_types = {'text': tf.int32}

        tgt_types = {'tgt_as':tf.int32}

        el_types = {'pid':tf.string,'tb_id0':tf.string,'tb_id1':tf.string}

        input_defaults = {'text': 0}

        tgt_defaults = {'tgt_as':0}

        el_defaults = {'pid':'-','tb_id0':'-','tb_id1':'-'}

        self._shapes_predict = input_shapes, tgt_shapes, el_shapes
        self._types_predict = input_types, tgt_types,el_types
        self._defaults_predict = input_defaults, tgt_defaults, el_defaults

    def shorten_tokenlist_if_necessary(self,tokenlist):
        if len(tokenlist)>self._max_token_length:
            #use a special token to mark the break, don't use self._tok_vocab_size+2 because in pre-training it was the mask token
            return tokenlist[:int(self._max_token_length/2)]+tokenlist[-int(self._max_token_length/2):]
        else:
            return tokenlist

    def _parse_fn(self, element):
        switch_order=self.bool_decision()
        if 'tgt' in element.keys():
            if switch_order:
                textblockone=element['textblock2']
                textblocktwo=element['textblock1']
            else:
                textblockone=element['textblock1']
                textblocktwo=element['textblock2']
            tar_as=[element['tgt']]
        else:
            take_from_same_article=self.bool_decision()
            if take_from_same_article:
                if switch_order:
                    textblockone=element['textblock2']
                    textblocktwo=element['textblock1']
                else:
                    textblockone=element['textblock1']
                    textblocktwo=element['textblock2']
                tar_as=[1]
            else:
                if switch_order:
                    textblockone=element['textblock2']
                    textblocktwo=element['othertextblock']
                else:
                    textblockone=element['textblock1']
                    textblocktwo=element['othertextblock']
                tar_as=[0]
        tokenlistone=self.shorten_tokenlist_if_necessary(self._tokenizer_de.encode(textblockone))
        tokenlisttwo=self.shorten_tokenlist_if_necessary(self._tokenizer_de.encode(textblocktwo))
        text_index_list=[self._tok_vocab_size]+tokenlistone+[self._tok_vocab_size+1]+tokenlisttwo+[self._tok_vocab_size+1]
        if self._flags.predict_mode:
            inputs = {'text':[text_index_list]}
            tgts = {'tgt_as': [tar_as]}
            return inputs, tgts, {'textblock1':textblockone,'textblock2':textblocktwo}
        inputs = {'text':text_index_list}
        tgts = {'tgt_as': tar_as}

        return inputs, tgts

    def generateDataSetPredict(self):

        dataset = tf.data.Dataset.from_generator(self.generator_fn_predict2,
                                                     output_shapes=self._shapes_predict, output_types=self._types_predict)
        dataset = (dataset
                       .padded_batch(self._flags.val_batch_size, self._shapes_predict, self._defaults_predict)
                       .prefetch(1))

        return dataset

    def get_input_fn_predict2(self,fname):
        self._mode = 'predict'
        self._fnames = [fname]
        if fname.endswith('json'):
            with open(fname, 'r',encoding="utf-8") as f:
                self._worklist = json.load(f)

        return self.generateDataSetPredict()

    def _parse_fn_predict(self, element):

        dummy_target=[1]
        textblockone=element['text1']
        textblocktwo=element['text2']
        tokenlistone=self.shorten_tokenlist_if_necessary(self._tokenizer_de.encode(textblockone))
        tokenlisttwo=self.shorten_tokenlist_if_necessary(self._tokenizer_de.encode(textblocktwo))
        text_index_list=[self._tok_vocab_size]+tokenlistone+[self._tok_vocab_size+1]+tokenlisttwo+[self._tok_vocab_size+1]
        inputs = {'text':text_index_list}
        tgts = {'tgt_as': dummy_target}
        elems ={'pid':[element['pid']],'tb_id0':[element['tb_id0']],'tb_id1':[element['tb_id1']]}
        return inputs, tgts, elems

    def get_textblock_from_another_article(self,art_id,articlelist):
        cur_art_id=art_id
        while cur_art_id==art_id:
            element=random.choice(articlelist)
            cur_art_id=element['art_id']
        if self.bool_decision():
            return element['textblock1']
        else:
            return element['textblock2']

    def generator_fn(self):
        random.shuffle(self._worklist)
        for element in self._worklist:
            if 'tgt' not in element.keys():
                element['othertextblock']=self.get_textblock_from_another_article(element['art_id'],self._worklist)
            yield self._parse_fn(element)

    def generator_fn_predict(self,fname):
            with open(fname, 'r',encoding="utf-8") as f:
                articlelist = json.load(f)
                random.shuffle(articlelist)
                for element in articlelist:
                    if 'tgt' not in element.keys():
                        element['othertextblock']=self.get_textblock_from_another_article(element['art_id'],articlelist)
                    yield self._parse_fn(element)

    def generator_fn_predict2(self):
            nsplist = self._worklist
            predictlist=[]
            for page in nsplist['page']:
                textblocklist=[]
                for article in page['articles']:
                    for text_block in article['text_blocks']:
                        textblocklist.append({'tbid':text_block['text_block_id'],'text':text_block['text'].replace('\n',' ')})
                for i in range(len(textblocklist)-1):
                    for j in range(len(textblocklist)-i-1):
                        predictlist.append({'pid':page['page_file'],'tb_id0':textblocklist[i]['tbid'],'text1':textblocklist[i]['text'],'tb_id1':textblocklist[j+i+1]['tbid'],'text2':textblocklist[j+i+1]['text']})
            for element in predictlist:
                yield self._parse_fn_predict(element)

    def bool_decision(self):
        return random.choice([True, False])






