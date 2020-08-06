import tensorflow as tf
import model_fn.model_fn_nlp.util_nlp.graphs_bert_sop as graphs
from model_fn.model_fn_base import ModelBase

class ModelBertNSP(ModelBase):
    def __init__(self, params):
        super(ModelBertNSP, self).__init__(params)
        self.flags = self._params['flags']
        self._graph = self.get_graph()
        self._vocab_size = params['tok_size']
        self.metrics["eval"]["acc_mlm"] = tf.keras.metrics.Accuracy()
        self.metrics["train"]["acc_mlm"] = tf.keras.metrics.Accuracy()
        self.metrics["eval"]["acc_nsp"] = tf.keras.metrics.Accuracy()
        self.metrics["train"]["acc_nsp"] = tf.keras.metrics.Accuracy()

    def get_graph(self):
        return getattr(graphs, self._params['flags'].graph)(self._params)

    def get_predictions(self):
        return {
            "mlm_pred_ids": self._graph_out['mlm_pred_ids'],
            "nsp_pred_ids": self._graph_out['nsp_pred_ids'],
            "masked_index" : self._graph_out['masked_index'],
            "mlm_logits": self._graph_out['mlm_logits'],
            "nsp_logits": self._graph_out['nsp_logits'],
        }

    def get_placeholder(self):
        ret = {"text": tf.compat.v1.placeholder(tf.int32, [None,None], name="inputText"),
                "masked_index": tf.compat.v1.placeholder(tf.int32, [None,None], name="inputTextlength"),
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
        tags_mlm = self._targets['tgt_mlm']
        tags_nsp = self._targets['tgt_nsp']
        tags_nsp = tags_nsp[:, 0]
        nsp_logits=tf.reduce_mean(tf.transpose(self._graph_out['nsp_logits'],[0,2,1]),axis=-1)
        masked_index=tf.cast(self._graph_out['masked_index'],tf.bool)
        mlm_loss=tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=tags_mlm, logits=self._graph_out['mlm_logits'],weights=masked_index)#, reduction=tf.compat.v1.losses.Reduction.MEAN)
        nsp_loss=tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=tags_nsp, logits=nsp_logits)#, reduction=tf.compat.v1.losses.Reduction.MEAN)
        return mlm_loss+nsp_loss

    def loss(self, predictions, targets):
        tags_mlm = targets['tgt_mlm']
        tags_nsp = targets['tgt_nsp']
        tags_nsp = tags_nsp[:, 0]
        nsp_logits=tf.reduce_mean(tf.transpose(predictions['nsp_logits'],[0,2,1]),axis=-1)
        masked_index=tf.cast(predictions['masked_index'],tf.bool)
        mlm_loss=tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=tags_mlm, logits=predictions['mlm_logits'],weights=masked_index)#, reduction=tf.compat.v1.losses.Reduction.MEAN)
        nsp_loss=tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=tags_nsp, logits=nsp_logits)#, reduction=tf.compat.v1.losses.Reduction.MEAN)
        return mlm_loss+nsp_loss

    def get_call_graph_signature(self):
        call_graph_signature = [{'text':tf.TensorSpec(shape=[None,None], dtype=tf.int32),
                                 'masked_index':tf.TensorSpec(shape=[None, None], dtype=tf.int32)},
                                {'tgt_mlm': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                                 'tgt_nsp': tf.TensorSpec(shape=[None, None], dtype=tf.int32)}]
        return call_graph_signature

    def to_tensorboard(self, graph_out_dict, targets, input_features):
        weights=tf.cast(graph_out_dict['masked_index'],tf.bool)
        self.metrics[self._mode]["loss"](graph_out_dict["loss"])
        tgt_mlm=targets['tgt_mlm']
        tgt_nsp=targets['tgt_nsp']
        mlm_pred_ids=graph_out_dict['mlm_pred_ids']
        nsp_pred_ids=tf.cast(tf.round(tf.reduce_mean(tf.cast(graph_out_dict['nsp_pred_ids'],tf.float32),axis=-1)),tf.int32)
        self.metrics[self._mode]["acc_mlm"].update_state(tgt_mlm, mlm_pred_ids, weights)
        self.metrics[self._mode]["acc_nsp"].update_state(tgt_nsp[:, 0], nsp_pred_ids)





