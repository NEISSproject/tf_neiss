from trainer.trainer_base import TrainerBase
import model_fn.model_fn_nlp.model_fn_as as models
import util.flags as flags
from input_fn.input_fn_nlp.input_fn_as import InputFnAS
import json
import tensorflow as tf
from os.path import join, basename

# Model parameter
# ===============
flags.define_string('model_type', 'ModelAS', 'Model Type to use choose from: ModelTriangle')
flags.define_string('tags', 'ner_uja.txt', 'path to tag vocabulary')
flags.define_string('tokenizer', '../../../data/tokenizer/tigertokenizer_de', 'path to tokenizer for encoder')
flags.define_string('bert_dir', '../../../bertmodels/bertdeminiwu', 'path to pretrained bert_model checkpoint')
flags.define_string('bert_graph', 'BERTMini', 'class name of pretrained bert graph architecture')
flags.define_integer('max_tokens_text_part', 80,
                     'maximal number of tokens in a text part of the input function')
flags.define_string('graph', 'NERwithMiniBERT', 'class name of graph architecture')
flags.define_list('add_types', str, 'types that are add features int or float',
                  "", [])
flags.define_integer('buffer', 1,
                     'number of training samples hold in the cache. (effects shuffling)')
flags.define_boolean('predict_mode', False, 'If and only if true the prediction will be accomplished')
flags.define_boolean('predict_features', True, 'If and only if true the prediction is for page_files')
flags.define_string('predict_list', None, '.lst-file specifying the dataset used for prediction')
flags.define_string('predict_dir', '', 'path/to/file where to write the prediction')
flags.define_integer('max_tb_per_page_prediction', 250,
                     'defines the maximal allowed number of textblocks on a single page, such that a prediction will be made for this page')
flags.FLAGS.parse_flags()


class TrainerAS(TrainerBase):
    def __init__(self):
        super(TrainerAS, self).__init__()
        self._input_fn_generator = InputFnAS(self._flags)
        self._input_fn_generator.print_params()
        self._params['tok_size'] = self._input_fn_generator._tok_vocab_size+3
        self._model_class = getattr(models, self._flags.model_type)
        # self._graph.info()

    def predict(self):

        print("Predict:")
        if not self._model:
            self._model = self._model_class(self._params)
        if not self._model.graph_eval:
            self._model.graph_eval = self._model.get_graph()
        if not self._checkpoint_obj_val:
            self._checkpoint_obj_val = tf.train.Checkpoint(model=self._model.graph_eval)

        self._checkpoint_obj_val.restore(tf.train.latest_checkpoint(self._flags.checkpoint_dir))
        call_graph_signature = self._model.get_call_graph_signature()

        @tf.function(input_signature=call_graph_signature)
        def call_graph(input_features_, targets_):
            self._model.graph_eval._graph_out = self._model.graph_eval(input_features_, training=False)
            self._model.graph_eval._graph_out['probs']=tf.reduce_mean(tf.transpose(self._model.graph_eval._graph_out['probabilities'],[0,2,1]),axis=-1)
            return self._model.graph_eval._graph_out


        self._input_fn_generator._mode = 'predict'
        self._fnames = []
        with open(self._flags.predict_list, 'r') as f:
            self._fnames.extend(f.read().splitlines())
        for fname in self._fnames:
            prediction_list=[]
            for (input_features, targets, input_element) in self._input_fn_generator.generator_fn_predict(fname):
                input_features['text']=tf.cast(input_features['text'],tf.int32)
                pred_out_dict = call_graph(input_features, targets)
                input_element['tgt']=targets['tgt_as'][0][0]
                input_element['prob']=str(pred_out_dict['probs'][0][1].numpy())
                prediction_list.append(input_element)
            with open(join(self._flags.predict_dir,'pred_'+basename(fname)),'w+') as g:
                json.dump(prediction_list,g)


    def predict2_batch(self):

        print("Predict:")
        if not self._model:
            self._model = self._model_class(self._params)
        if not self._model.graph_eval:
            self._model.graph_eval = self._model.get_graph()
        if not self._checkpoint_obj_val:
            self._checkpoint_obj_val = tf.train.Checkpoint(model=self._model.graph_eval)

        self._checkpoint_obj_val.restore(tf.train.latest_checkpoint(self._flags.checkpoint_dir))
        call_graph_signature = self._model.get_call_graph_signature()

        @tf.function(input_signature=call_graph_signature)
        def call_graph(input_features_, targets_):
            self._model.graph_eval._graph_out = self._model.graph_eval(input_features_, training=False)
            self._model.graph_eval._graph_out['probs']=tf.reduce_mean(tf.transpose(self._model.graph_eval._graph_out['probabilities'],[0,2,1]),axis=-1)
            return self._model.graph_eval._graph_out


        self._input_fn_generator._mode = 'predict'
        self._fnames = []
        with open(self._flags.predict_list, 'r') as f:
            self._fnames.extend(f.read().splitlines())
        for fname in self._fnames:
            pagedic={}
            index=0
            prob_default=[0.5]
            import time
            start= time.time()
            with open(fname, 'r',encoding="utf-8") as f:
                raw_data = json.load(f)
            predictlist=[]
            for page in raw_data['page']:
                textblocklist=[]
                for article in page['articles']:
                    for text_block in article['text_blocks']:
                        textblocklist.append({'tbid':text_block['text_block_id'],'text':text_block['text'].replace('\n',' ')})
                if len(textblocklist)<= self._flags.max_tb_per_page_prediction:
                    for i in range(len(textblocklist)-1):
                        for j in range(len(textblocklist)-i-1):
                            predictlist.append({'pid':page['page_file'],'tb_id0':textblocklist[i]['tbid'],'text1':textblocklist[i]['text'],'tb_id1':textblocklist[j+i+1]['tbid'],'text2':textblocklist[j+i+1]['text']})
                else:
                    pagedic[page['page_file']]={'edge_features':{'default':prob_default}}


            for (input_features, targets, input_element) in self._input_fn_generator.get_input_fn_predict2(predictlist):

                input_features['text']=tf.cast(input_features['text'],tf.int32)
                pred_out_dict = call_graph(input_features, targets)

                for i in range(len(input_element['pid'])):
                    index+=1
                    pid=input_element['pid'][i].numpy().item().decode('utf-8')
                    tb_id0=input_element['tb_id0'][i].numpy().item().decode('utf-8')
                    tb_id1=input_element['tb_id1'][i].numpy().item().decode('utf-8')
                    prob=[pred_out_dict['probs'][i][1].numpy().item()]
                    if pid in pagedic.keys():
                        if tb_id0 in pagedic[pid]['edge_features'].keys():
                            pagedic[pid]['edge_features'][tb_id0][tb_id1]=prob
                        else:
                            pagedic[pid]['edge_features'][tb_id0]={tb_id1:prob}
                        if tb_id1 in pagedic[pid]['edge_features'].keys():
                            pagedic[pid]['edge_features'][tb_id1][tb_id0]=prob
                        else:
                            pagedic[pid]['edge_features'][tb_id1]={tb_id0:prob}
                    else:
                        pagedic[pid]={'edge_features':{tb_id0:{tb_id1:prob},
                                               tb_id1:{tb_id0:prob}}}
                    if index%10000==0:
                        ende = time.time()
                        print(index,ende-start)
                        start= time.time()
            with open(join(self._flags.predict_dir,'pred_'+basename(fname)),'w+') as g:
                json.dump(pagedic,g,indent=4)



if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    trainer = TrainerAS()
    if flags.FLAGS.predict_mode :
        if flags.FLAGS.predict_features:
            trainer.predict2_batch()
        else:
            trainer.predict()
    else:
        trainer.train()
