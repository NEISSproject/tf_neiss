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
flags.define_string('graph', 'NERwithMiniBERT', 'class name of graph architecture')
flags.define_list('add_types', str, 'types that are add features int or float',
                  "", [])
flags.define_integer('buffer', 1,
                     'number of training samples hold in the cache. (effects shuffling)')
flags.define_boolean('predict_mode', False, 'If and only if true the prediction will be accomplished')
flags.define_string('predict_list', None, '.lst-file specifying the dataset used for prediction')
flags.define_string('predict_dir', '', 'path/to/file where to write the prediction')
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
            for (input_features, targets) in self._input_fn_generator.generator_fn_predict(fname):
                input_features['text']=tf.cast(input_features['text'],tf.int32)
                pred_out_dict = call_graph(input_features, targets)
                prediction_list.append({'tgt':targets['tgt_as'][0][0],'prob':str(pred_out_dict['probs'][0][1].numpy())})
            with open(join(self._flags.predict_dir,'pred_'+basename(fname)),'w+') as g:
                json.dump(prediction_list,g)



if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    trainer = TrainerAS()
    if flags.FLAGS.predict_mode :
        trainer.predict()
    else:
        trainer.train()
