import tensorflow as tf
import model_fn.model_fn_nlp.util_nlp.graphs_bert_sop as graphs
from model_fn.model_fn_base import ModelBase

class ModelBertSOP(ModelBase):
    def __init__(self, params):
        super(ModelBertSOP, self).__init__(params)
        self.flags = self._params['flags']
        self._graph = self.get_graph()
        self._vocab_size = params['tok_size']
        self.metrics["eval"]["acc_mlm"] = tf.keras.metrics.Accuracy()
        self.metrics["train"]["acc_mlm"] = tf.keras.metrics.Accuracy()
        self.metrics["eval"]["acc_so"] = tf.keras.metrics.Accuracy()
        self.metrics["train"]["acc_so"] = tf.keras.metrics.Accuracy()

    def get_graph(self):
        return getattr(graphs, self._params['flags'].graph)(self._params)

    def get_predictions(self):
        return {
            "mlm_pred_ids": self._graph_out['mlm_pred_ids'],
            "sop_pred_ids": self._graph_out['sop_pred_ids'],
            "masked_index" : self._graph_out['masked_index'],
            "mlm_logits": self._graph_out['mlm_logits'],
            "sop_logits": self._graph_out['sop_logits'],
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
        tags_sop = self._targets['tgt_sop']
        tags_sop = tags_sop[:, 0]
        sop_logits=tf.reduce_mean(tf.transpose(self._graph_out['sop_logits'],[0,2,1]),axis=-1)
        masked_index=tf.cast(self._graph_out['masked_index'],tf.bool)
        mlm_loss=tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=tags_mlm, logits=self._graph_out['mlm_logits'],weights=masked_index)#, reduction=tf.compat.v1.losses.Reduction.MEAN)
        sop_loss=tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=tags_sop, logits=sop_logits)#, reduction=tf.compat.v1.losses.Reduction.MEAN)
        return mlm_loss+sop_loss

    def loss(self, predictions, targets):
        tags_mlm = targets['tgt_mlm']
        tags_sop = targets['tgt_sop']
        tags_sop = tags_sop[:, 0]
        sop_logits=tf.reduce_mean(tf.transpose(predictions['sop_logits'],[0,2,1]),axis=-1)
        masked_index=tf.cast(predictions['masked_index'],tf.bool)
        mlm_loss=tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=tags_mlm, logits=predictions['mlm_logits'],weights=masked_index)#, reduction=tf.compat.v1.losses.Reduction.MEAN)
        sop_loss=tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=tags_sop, logits=sop_logits)#, reduction=tf.compat.v1.losses.Reduction.MEAN)
        return mlm_loss+sop_loss

    def get_call_graph_signature(self):
        call_graph_signature = [{'text':tf.TensorSpec(shape=[None,None], dtype=tf.int32),
                                 'masked_index':tf.TensorSpec(shape=[None, None], dtype=tf.int32)},
                                {'tgt_mlm': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                                 'tgt_sop': tf.TensorSpec(shape=[None, None], dtype=tf.int32)}]
        return call_graph_signature

    def to_tensorboard(self, graph_out_dict, targets, input_features):
        weights=tf.cast(graph_out_dict['masked_index'],tf.bool)
        self.metrics[self._mode]["loss"](graph_out_dict["loss"])
        tgt_mlm=targets['tgt_mlm']
        tgt_sop=targets['tgt_sop']
        mlm_pred_ids=graph_out_dict['mlm_pred_ids']
        sop_pred_ids=tf.cast(tf.round(tf.reduce_mean(tf.cast(graph_out_dict['sop_pred_ids'],tf.float32),axis=-1)),tf.int32)
        self.metrics[self._mode]["acc_mlm"].update_state(tgt_mlm, mlm_pred_ids, weights)
        self.metrics[self._mode]["acc_so"].update_state(tgt_sop[:, 0], sop_pred_ids)

if __name__ == '__main__':
    def get_loss(targets,probs,masked_index):
        one_hot_mlm=tf.cast(tf.one_hot(targets,6),tf.float32)
        per_example_loss = -tf.reduce_sum(probs * one_hot_mlm, axis=[-1])
        numerator = tf.reduce_sum(tf.cast(masked_index,tf.float32) * per_example_loss)
        denominator = tf.reduce_sum(tf.cast(masked_index,tf.float32)) + 1e-5
        mlm_loss = tf.cast(1.0,tf.float32)+numerator / denominator
        one_hot_so=tf.cast(tf.one_hot(targets[:,0],2),tf.float32)
        per_example_loss_so = -tf.reduce_sum(probs[:,0,0:2] * one_hot_so, axis=[-1])
        numerator_so = tf.reduce_sum(per_example_loss_so)
        denominator_so = tf.cast(tf.shape(targets)[0],tf.float32)
        so_loss=tf.cast(1.0,tf.float32)+numerator_so / denominator_so
        print(mlm_loss)
        print(so_loss)
    targets=tf.cast([[1,2,3,4,5],[0,5,4,3,2],[1,1,1,1,2]],tf.int32)
    masked_index=tf.cast([[0,1,0,0,1],[0,1,0,1,0],[0,0,1,1,0]],tf.int32)
    probs=tf.cast([[
        [0.2,0.8,0,0,0,0],[0,0,1.0,0,0,0],[0,0,0,1.0,0,0],[0,0,0,0,1.0,0],[0,0,0,0,0,1.0]
    ],[
        [0.2,0.8,0,0,0,0],[0,0,0,0,0,1.0],[0,0,0,0,1.0,0],[0,0,0,1.0,0,0],[0,0,1.0,0,0,0]
    ],[
        [0.2,0.8,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]
    ]],tf.float32)
    #get_loss(targets,probs,masked_index)
    print(targets[:,0])



