import tensorflow as tf
import model_fn.model_fn_nlp.util_nlp.graphs_bert_lm as graphs
from model_fn.model_fn_base import ModelBase

class ModelBertLM(ModelBase):
    def __init__(self, params):
        super(ModelBertLM, self).__init__(params)
        self.flags = self._params['flags']
        self._graph = self.get_graph()
        self.metrics["eval"]["accuracy"] = tf.keras.metrics.Accuracy()
        self.metrics["train"]["accuracy"] = tf.keras.metrics.Accuracy()

    def get_graph(self):
        return getattr(graphs, self._params['flags'].graph)(self._params)

    def get_predictions(self):
        return {
            "classes": self._graph_out['pred_ids'],
            #"probabilities": self._graph_out['probabilities'],
            "sentencelength": self._graph_out['sentencelength'],
            "logits": self._graph_out['logits'],
            'masked_index': self._graph_out['masked_index'],
        }

    def get_placeholder(self):
        ret = {"sentence": tf.compat.v1.placeholder(tf.int32, [None,None], name="inputSentence"),
                "sentencelength": tf.compat.v1.placeholder(tf.int32, [None,None], name="inputSentencelength"),
               "masked_index": tf.compat.v1.placeholder(tf.int32, [None,None], name="inputMaskedIndex"),
                }
        for idx, _ in enumerate(self.flags.add_types):
            ret['feat{}'.format(idx)] = tf.compat.v1.placeholder(tf.int32, [None, None], name='feature{}'.format(idx))
        return ret

    def get_output_nodes(self, has_graph=True):
        if has_graph:
            #tf.identity(self._graph_out['probabilities'], name="probs")  # name to grab from java
            tf.identity(self._graph_out['pred_ids'], name="ids")  # name to grab from java
        return "ids"  # return names as comma separated string without spaces

    def get_loss(self):
        tags = self._targets['tgt']
        sentencelength=self._graph_out['sentencelength']
        masked_index=self._graph_out['masked_index']
        weights = tf.sequence_mask(sentencelength)
        new_weights=tf.cast(tf.multiply(tf.cast(weights,tf.int32),tf.cast(masked_index,tf.int32)),tf.bool)
        return tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=tags, logits=self._graph_out['logits'],weights=new_weights,reduction=tf.compat.v1.losses.Reduction.MEAN)

    def loss(self, predictions, targets):
        tags = targets['tgt']
        sentencelength=predictions['sentencelength']
        masked_index=predictions['masked_index']
        weights = tf.sequence_mask(sentencelength)
        new_weights=tf.cast(tf.multiply(tf.cast(weights,tf.int32),tf.cast(masked_index,tf.int32)),tf.bool)
        rvalue=tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=tags, logits=predictions['logits'],weights=new_weights)#, reduction=tf.compat.v1.losses.Reduction.MEAN)
        return rvalue

    def get_call_graph_signature(self):
        call_graph_signature = [{'sentence':tf.TensorSpec(shape=[None,None], dtype=tf.int32),
                                 'sentencelength':tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                                 'masked_index':tf.TensorSpec(shape=[None, None], dtype=tf.int32)},
                                {'tgt': tf.TensorSpec(shape=[None, None], dtype=tf.int32)}]
        return call_graph_signature

    def to_tensorboard(self, graph_out_dict, targets, input_features):
        sentencelength=graph_out_dict['sentencelength']
        masked_index=graph_out_dict['masked_index']
        weights = tf.sequence_mask(sentencelength)
        new_weights=tf.cast(tf.multiply(tf.cast(weights,tf.int32),tf.cast(masked_index,tf.int32)),tf.bool)
        self.metrics[self._mode]["loss"](graph_out_dict["loss"])
        self.metrics[self._mode]["accuracy"].update_state(targets['tgt'], graph_out_dict['pred_ids'], new_weights)


