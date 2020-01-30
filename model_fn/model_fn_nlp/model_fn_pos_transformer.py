import shutil
import numpy as np
import tensorflow as tf
import model_fn.model_fn_nlp.util_nlp.graphs_pos as graphs
from input_fn.input_fn_nlp.util_input_fn_nlp.StringMapper import get_sm
from model_fn.model_fn_base import ModelBase

class ModelTFPOS(ModelBase):
    def __init__(self, params):
        super(ModelTFPOS, self).__init__(params)
        # Indices of NOT dummy class
        self._pos_indices = []
        self.flags = self._params['flags']

        self._tag_string_mapper = get_sm(self._params['flags'].tags)
        for id in range(self._tag_string_mapper.size()):

            tag = self._tag_string_mapper.get_value(id)
            if tag.strip() != 'O':
                self._pos_indices.append(id)

        self._graph = self.get_graph()
        self.metrics["eval"]["accuracy"] = tf.keras.metrics.Accuracy()
        self.metrics["train"]["accuracy"] = tf.keras.metrics.Accuracy()
        self._keras_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction='none')
        self._eval_keras_loss_metric = tf.keras.metrics.Mean('eval_keras_loss', dtype=tf.float32)

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
            "probabilities": self._graph_out['probabilities'],
            "sentencelength": self._graph_out['sentencelength']
        }

    def get_placeholder(self):
        ret = {"sentence": tf.compat.v1.placeholder(tf.int32, [None, None, None], name="inputSentence"),
                "sentencelength": tf.compat.v1.placeholder(tf.int32, [None], name="inputSentencelength"),
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
        sentencelength=self._graph_out['sentencelength']
        weights = tf.sequence_mask(sentencelength)
        return tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=tags, logits=self._graph_out['logits'],weights=weights,reduction=tf.compat.v1.losses.Reduction.MEAN)

    def loss(self, predictions, targets):
        tags = targets['tgt']
        sentencelength=predictions['sentencelength']
        #---old<<
        weights = tf.sequence_mask(sentencelength)
        rvalue=tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=tags, logits=predictions['logits'],weights=weights)#, reduction=tf.compat.v1.losses.Reduction.MEAN)
        #--new<<
        #loss_ = self._keras_loss(tags,predictions['logits'])
        #mask=tf.cast(tf.sequence_mask(sentencelength),tf.float32)
        #rvalue=loss_*mask
        #--new>>
        return rvalue

    def keras_loss(self, predictions, targets):
        tags = targets['tgt']
        sentencelength=predictions['sentencelength']
        dims=tf.shape(predictions['pred_ids'])
        recip=tf.math.reciprocal(tf.cast(sentencelength,tf.float32))*tf.cast(dims[-1],tf.float32)
        exp_seq=tf.broadcast_to(tf.expand_dims(recip,axis=-1), dims)
        weights = tf.math.multiply(tf.cast(tf.sequence_mask(sentencelength,dims[-1]),tf.float32),exp_seq)
        rvalue=self._keras_loss(tags,predictions['logits'],weights)
        #Each weight has to multiplied with maxlen/seq_length
        return rvalue

    def get_call_graph_signature(self):
        call_graph_signature = [{'sentence':tf.TensorSpec(shape=[None,None], dtype=tf.int32),
                                 'sentencelength':tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                                 'tar_inp':tf.TensorSpec(shape=[None, None], dtype=tf.int32)},
                                {'tgt': tf.TensorSpec(shape=[None, None], dtype=tf.int32)}]
        return call_graph_signature

    def to_tensorboard(self, graph_out_dict, targets, input_features):
        self.metrics[self._mode]["loss"](graph_out_dict["loss"])
        self.metrics[self._mode]["accuracy"].update_state(targets['tgt'], graph_out_dict['pred_ids'], tf.sequence_mask(graph_out_dict['sentencelength']))


if __name__ == "__main__":
    string_mapper=get_sm('../../tests/workdir_pos/stts_tiger.txt')
    pos_indices = []
    for id in range(string_mapper.size()):
        tag = string_mapper.get_value(id)
        if tag.strip() != 'O':
            pos_indices.append(id)
    print(pos_indices,string_mapper.get_value(56))
