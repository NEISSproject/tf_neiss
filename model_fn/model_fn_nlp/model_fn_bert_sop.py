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
            "pred_ids": self._graph_out['pred_ids'],
            "probs": self._graph_out['probs'],
            "masked_index" : self._graph_out['masked_index'],
            "logits": self._graph_out['logits'],
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
        targets = self._targets['tgt']
        probs=self._graph_out['probs']
        masked_index=self._graph_out['masked_index']
        one_hot_mlm=tf.one_hot(targets,self._vocab_size)
        per_example_loss = -tf.reduce_sum(probs * one_hot_mlm, axis=[-1])
        numerator = tf.reduce_sum(tf.cast(masked_index,tf.float32) * per_example_loss)
        denominator = tf.reduce_sum(tf.cast(masked_index,tf.float32)) + 1e-5
        mlm_loss = tf.cast(1.0,tf.float32)+numerator / denominator
        one_hot_so=tf.cast(tf.one_hot(targets[:,0],2),tf.float32)
        per_example_loss_so = -tf.reduce_sum(probs[:,0,0:2] * one_hot_so, axis=[-1])
        numerator_so = tf.reduce_sum(per_example_loss_so)
        denominator_so = tf.cast(tf.shape(targets)[0],tf.float32)
        so_loss=tf.cast(1.0,tf.float32)+numerator_so / denominator_so
        return mlm_loss+so_loss

    def loss(self, predictions, targets):
        targets = targets['tgt']
        probs=predictions['probs']
        masked_index=predictions['masked_index']
        one_hot_mlm=tf.one_hot(targets,self._vocab_size)
        per_example_loss = -tf.reduce_sum(probs * one_hot_mlm, axis=[-1])
        numerator = tf.reduce_sum(tf.cast(masked_index,tf.float32) * per_example_loss)
        denominator = tf.reduce_sum(tf.cast(masked_index,tf.float32)) + 1e-5
        mlm_loss = tf.cast(1.0,tf.float32)+numerator / denominator
        one_hot_so=tf.cast(tf.one_hot(targets[:,0],2),tf.float32)
        per_example_loss_so = -tf.reduce_sum(probs[:,0,0:2] * one_hot_so, axis=[-1])
        numerator_so = tf.reduce_sum(per_example_loss_so)
        denominator_so = tf.cast(tf.shape(targets)[0],tf.float32)
        so_loss=tf.cast(1.0,tf.float32)+numerator_so / denominator_so
        return mlm_loss+so_loss

    def get_call_graph_signature(self):
        call_graph_signature = [{'text':tf.TensorSpec(shape=[None,None], dtype=tf.int32),
                                 'masked_index':tf.TensorSpec(shape=[None, None], dtype=tf.int32)},
                                {'tgt': tf.TensorSpec(shape=[None, None], dtype=tf.int32)}]
        return call_graph_signature

    def to_tensorboard(self, graph_out_dict, targets, input_features):
        weights=tf.cast(graph_out_dict['masked_index'],tf.bool)
        self.metrics[self._mode]["loss"](graph_out_dict["loss"])
        tgt=targets['tgt']
        pred_ids=graph_out_dict['pred_ids']
        self.metrics[self._mode]["acc_mlm"].update_state(tgt, pred_ids, weights)
        self.metrics[self._mode]["acc_so"].update_state(tgt[:,0], pred_ids[:,0])

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
    get_loss(targets,probs,masked_index)



