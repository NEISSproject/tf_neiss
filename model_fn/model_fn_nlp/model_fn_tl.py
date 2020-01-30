import tensorflow as tf
import model_fn.model_fn_nlp.util_nlp.graphs_tl as graphs
from model_fn.model_fn_base import ModelBase


class ModelTL(ModelBase):
    def __init__(self, params):
        super(ModelTL, self).__init__(params)
        # Indices of NOT dummy class
        self._pos_indices = []
        self.flags = self._params['flags']

        self._graph = self.get_graph()
        self._loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        self.metrics["eval"]["accuracy"] = tf.keras.metrics.SparseCategoricalAccuracy()
        self.metrics["train"]["accuracy"] = tf.keras.metrics.SparseCategoricalAccuracy()
        self._targets = None

    def get_graph(self):
        return getattr(graphs, self._params['flags'].graph)(self._params)

    def get_predictions(self):
        return {
            "classes": self._graph_out['pred_ids'],
            "probabilities": self._graph_out['probabilities']
        }

    def get_output_nodes(self, has_graph=True):
        if has_graph:
            tf.identity(self._graph_out['probabilities'], name="probs")  # name to grab from java
            tf.identity(self._graph_out['pred_ids'], name="ids")  # name to grab from java
        return "logProbs,ids"  # return names as comma separated string without spaces

    def get_loss(self):
        pred = self._graph_out['logits']
        real = self._targets
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self._loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def loss(self, predictions, targets):
        pred = predictions['logits']
        real = targets
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self._loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def to_tensorboard(self, graph_out_dict, targets, input_features):
        self.metrics[self._mode]["loss"](graph_out_dict["loss"])
        self.metrics[self._mode]["accuracy"](targets, graph_out_dict['logits'])

    def get_call_graph_signature(self):
        call_graph_signature = [
            {'inputs':tf.TensorSpec(shape=(None, None), dtype=tf.int64),'tar_inp':tf.TensorSpec(shape=(None, None), dtype=tf.int64)},
            tf.TensorSpec(shape=(None, None), dtype=tf.int64)]
        return call_graph_signature
