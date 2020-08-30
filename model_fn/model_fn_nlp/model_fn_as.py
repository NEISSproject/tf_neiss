import tensorflow as tf
import model_fn.model_fn_nlp.util_nlp.graphs_as as graphs
from model_fn.model_fn_base import ModelBase
import logging

logger = logging.getLogger(__name__)

class ModelAS(ModelBase):
    def __init__(self, params):
        super(ModelAS, self).__init__(params)
        self.flags = self._params['flags']
        self._graph = self.get_graph()
        self._vocab_size = params['tok_size']
        self.metrics["eval"]["accuracy"] = tf.keras.metrics.Accuracy()
        self.metrics["train"]["accuracy"] = tf.keras.metrics.Accuracy()
        self.metrics["eval"]["precision"] = tf.keras.metrics.Precision()
        self.metrics["train"]["precision"] = tf.keras.metrics.Precision()
        self.metrics["eval"]["recall"] = tf.keras.metrics.Recall()
        self.metrics["train"]["recall"] = tf.keras.metrics.Recall()
        self.metrics["eval"]["roc"] = tf.keras.metrics.AUC()
        self.metrics["train"]["roc"] = tf.keras.metrics.AUC()

    def get_graph(self):
        return getattr(graphs, self._params['flags'].graph)(self._params)

    def get_predictions(self):
        return {
            "pred_ids": self._graph_out['pred_ids'],
            "probabilities": self._graph_out['probabilities'],
            "logits": self._graph_out['logits'],
        }

    def get_placeholder(self):
        ret = {"text": tf.compat.v1.placeholder(tf.int32, [None,None], name="inputText"),
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
        tags_as = self._targets['tgt_as']
        tags_as = tags_as[:, 0]
        as_logits=tf.reduce_mean(tf.transpose(self._graph_out['logits'],[0,2,1]),axis=-1)
        as_loss=tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=tags_as, logits=as_logits)#, reduction=tf.compat.v1.losses.Reduction.MEAN)
        return as_loss

    def loss(self, predictions, targets):
        tags_as = targets['tgt_as']
        tags_as = tags_as[:, 0]
        as_logits=tf.reduce_mean(tf.transpose(predictions['logits'],[0,2,1]),axis=-1)
        as_loss=tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=tags_as, logits=as_logits)#, reduction=tf.compat.v1.losses.Reduction.MEAN)
        return as_loss

    def get_call_graph_signature(self):
        call_graph_signature = [{'text':tf.TensorSpec(shape=[None,None], dtype=tf.int32)},
                                {'tgt_as': tf.TensorSpec(shape=[None, None], dtype=tf.int32)}]
        return call_graph_signature

    def to_tensorboard(self, graph_out_dict, targets, input_features):
        self.metrics[self._mode]["loss"](graph_out_dict["loss"])
        tgt_as=targets['tgt_as']
        pred_ids=tf.cast(tf.round(tf.reduce_mean(tf.cast(graph_out_dict['pred_ids'],tf.float32),axis=-1)),tf.int32)
        self.metrics[self._mode]["accuracy"].update_state(tgt_as[:, 0], pred_ids)
        self.metrics[self._mode]["precision"].update_state(tgt_as[:, 0], pred_ids)
        self.metrics[self._mode]["recall"].update_state(tgt_as[:, 0], pred_ids)
        probs=tf.reduce_mean(tf.transpose(graph_out_dict['probabilities'],[0,2,1]),axis=-1)
        self.metrics[self._mode]["roc"].update_state(tgt_as[:, 0], probs[:, 1])


    def write_tensorboard(self):
        """Write metrics to tensorboard-file (it's called after each epoch) and reset tf.keras.metrics"""
        with self.summary_writer[self._mode].as_default():
            if self._mode_training:
                tf.summary.scalar("learning_rate",
                                  self.custom_optimizer.get_current_learning_rate(self.optimizer.iterations - 1),
                                  step=self.optimizer.iterations - 1)
                tf.summary.scalar("learning_rate", self.custom_optimizer.get_current_learning_rate(self.optimizer.iterations),
                                  step=self.optimizer.iterations)
            else:
                # add
                pass

            precision=self.metrics[self._mode]['precision'].result()
            recall=self.metrics[self._mode]['recall'].result()
            tf.summary.scalar("f1",2*precision*recall/(precision+recall) , step=self.graph_train.global_epoch)
            logger.info("Reset all metics {}".format(self._mode))
            for metric in self.metrics[self._mode]:
                logger.debug("Write metric: {} with tf.name: {}".format(metric, self.metrics[self._mode][metric].name))
                tf.summary.scalar(metric, self.metrics[self._mode][metric].result(), step=self.graph_train.global_epoch)
                logger.debug("Reset metric: {} with tf.name: {}".format(metric, self.metrics[self._mode][metric].name))
                self.metrics[self._mode][metric].reset_states()







