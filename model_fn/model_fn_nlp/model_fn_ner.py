import shutil
import numpy as np
import tensorflow as tf
import model_fn.model_fn_nlp.util_nlp.graphs_ner as graphs
from input_fn.input_fn_nlp.util_input_fn_nlp.StringMapper import get_sm
from model_fn.model_fn_base import ModelBase

class ModelNER(ModelBase):
    def __init__(self, params):
        super(ModelNER, self).__init__(params)
        # Indices of NOT dummy class
        self.flags = self._params['flags']

        self._tag_string_mapper = get_sm(self._params['flags'].tags)
        self._graph = self.get_graph()
        self.metrics["eval"]["accuracy"] = tf.keras.metrics.Accuracy()
        self.metrics["train"]["accuracy"] = tf.keras.metrics.Accuracy()
        self.metrics["eval"]["precision"] = tf.keras.metrics.Precision()
        self.metrics["train"]["precision"] = tf.keras.metrics.Precision()
        self.metrics["eval"]["recall"] = tf.keras.metrics.Recall()
        self.metrics["train"]["recall"] = tf.keras.metrics.Recall()

    def get_graph(self):
        return getattr(graphs, self._params['flags'].graph)(self._params)

    def vocab_lookup(self, ids):
        l = []
        for b in ids:
            l2 = []
            for i in b:
                tag = self._tag_string_mapper.get_value(i)
                l2.append(tag)
            l.append(l2)

        return np.array(l, np.str)

    def get_predictions(self):
        return {
            "classes": self._graph_out['pred_ids'],
            "probabilities": self._graph_out['probabilities']
        }

    def get_placeholder(self):
        ret = {"sentence": tf.compat.v1.placeholder(tf.int32, [None, None, None], name="inputSentence")
                }
        for idx, _ in enumerate(self.flags.add_types):
            ret['feat{}'.format(idx)] = tf.compat.v1.placeholder(tf.int32, [None, None], name='feature{}'.format(idx))
        return ret

    def get_output_nodes(self, has_graph=True):
        if has_graph:
            tf.identity(self._graph_out['probabilities'], name="probs")  # name to grab from java
            tf.identity(self._graph_out['pred_ids'], name="ids")  # name to grab from java
        return "logProbs,ids"  # return names as comma separated string without spaces

    def get_loss(self):
        tags = self._targets['tgt']
        weights=tf.cast(self._targets['targetmask'],tf.bool)
        return tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=tags, logits=self._graph_out['logits'],weights=weights,reduction=tf.compat.v1.losses.Reduction.MEAN)

    def loss(self, predictions, targets):
        tags = targets['tgt']
        weights=tf.cast(targets['targetmask'],tf.bool)
        rvalue=tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=tags, logits=predictions['logits'],weights=weights)#, reduction=tf.compat.v1.losses.Reduction.MEAN)
        return rvalue

    def get_call_graph_signature(self):
        call_graph_signature = [{'sentence':tf.TensorSpec(shape=[None,None], dtype=tf.int32)},
                                {'tgt': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                                 'targetmask':tf.TensorSpec(shape=[None, None], dtype=tf.int32)}]
        return call_graph_signature

    def to_tensorboard(self, graph_out_dict, targets, input_features):
        weights=tf.cast(targets['targetmask'],tf.bool)
        self.metrics[self._mode]["loss"](graph_out_dict["loss"])
        self.metrics[self._mode]["accuracy"].update_state(targets['tgt'], graph_out_dict['pred_ids'], weights)
        true_positiv_indexes=tf.cast(tf.raw_ops.Equal(x=targets['tgt'],y=graph_out_dict['pred_ids']),tf.int32)
        self.metrics[self._mode]["precision"].update_state(true_positiv_indexes, tf.cast(tf.raw_ops.NotEqual(x=graph_out_dict['pred_ids'],y=self._tag_string_mapper.get_oov_id()),tf.int32), weights)
        self.metrics[self._mode]["recall"].update_state(tf.cast(tf.raw_ops.NotEqual(x=targets['tgt'],y=self._tag_string_mapper.get_oov_id()),tf.int32), true_positiv_indexes, weights)

    def init_new_training(self):
        checkpoint_obj = tf.train.Checkpoint(model=self.graph_train._pretrained_bert)
        if tf.train.get_checkpoint_state(self._flags.bert_dir):
            print("load pretrained bert_model from bert checkpoint: {}".format(self._flags.bert_dir))
            checkpoint_obj.restore(tf.train.latest_checkpoint(self._flags.bert_dir)).expect_partial()

