from trainer.trainer_base import TrainerBase
import model_fn.model_fn_nlp.model_fn_ner as models
import util.flags as flags
import json
import tensorflow as tf
from input_fn.input_fn_nlp.input_fn_ner import InputFnNER
from os.path import join, basename

# Model parameter
# ===============
flags.define_string('model_type', 'ModelNER', 'Model Type to use choose from: ModelTriangle')
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


class TrainerNER(TrainerBase):
    def __init__(self):
        super(TrainerNER, self).__init__()
        self._input_fn_generator = InputFnNER(self._flags)
        self._input_fn_generator.print_params()
        self._params['num_tags'] = self._input_fn_generator.get_num_tags()
        self._params['tok_size'] = self._input_fn_generator._tok_vocab_size+3
        self._params['tags'] = self._input_fn_generator.getTagMapper()
        self._model_class = getattr(models, self._flags.model_type)
        # self._graph.info()

    def predict(self):

        def get_top_tag(idlist):
            tagdict={}
            for i in range(len(idlist)):
                if idlist[i]!=self._params['tags'].get_oov_id():
                    tag=self._params['tags'].get_value(idlist[i])
                    if tag in tagdict.keys():
                        tagdict[tag]+=1
                    else:
                        tagdict[tag]=1
            cur_max=0
            cur_tag='O'
            for tag in tagdict.keys():
                if tagdict[tag]>cur_max:
                    cur_max=tagdict[tag]
                    cur_tag=tag
            return cur_tag


        def integrate_prediction_into_train_data(training_data,pred_ids):
            j=0
            for i in range(len(training_data)):
                cur_decode_length=training_data[i][3]
                training_data[i][1]=get_top_tag(pred_ids[j:j+cur_decode_length])
                j+=cur_decode_length
            return training_data
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
            return self._model.graph_eval._graph_out


        self._input_fn_generator._mode = 'predict'
        self._fnames = []
        with open(self._flags.predict_list, 'r') as f:
            self._fnames.extend(f.read().splitlines())
        for fname in self._fnames:
            prediction_list=[]
            for (input_features, targets, training_data) in self._input_fn_generator.generator_fn_predict(fname):
                input_features['sentence']=tf.cast(input_features['sentence'],tf.int32)
                pred_out_dict = call_graph(input_features, targets)
                pred_ids= pred_out_dict["pred_ids"][0].numpy()[1:-1]
                predicted_data=integrate_prediction_into_train_data(training_data,pred_ids)
                prediction_list.append(predicted_data)
            with open(join(self._flags.predict_dir,'pred_'+basename(fname)),'w+') as g:
                json.dump(prediction_list,g)


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    trainer = TrainerNER()
    if flags.FLAGS.predict_mode :
        trainer.predict()
    else:
        trainer.train()
