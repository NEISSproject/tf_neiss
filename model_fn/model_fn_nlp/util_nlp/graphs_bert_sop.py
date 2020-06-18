import tensorflow as tf

from model_fn.graph_base import GraphBase
from model_fn.model_fn_nlp.util_nlp.transformer import Encoder,BERTMini

class BERTMiniSOP(GraphBase):
    def __init__(self, params):
        super(BERTMiniSOP, self).__init__(params)
        self._flags = params['flags']
        self.bert=BERTMini(params)
        self._tracked_layers["last_layer_mlm"] = tf.keras.layers.Dense(params['tok_size'])
        self._tracked_layers["last_layer_so"] = tf.keras.layers.Dense(2)

    def call(self, inputs, training=None, mask=None):
        bert_out = self.bert(inputs)

        mlm_logits = self._tracked_layers["last_layer_mlm"](bert_out['enc_output'])  # (batch_size, tar_seq_len, target_vocab_size)
        sop_logits = self._tracked_layers["last_layer_so"](bert_out['enc_output'])  # (batch_size, tar_seq_len, 2)
        mlm_pred_ids = tf.argmax(input=mlm_logits, axis=2, output_type=tf.int32)
        sop_pred_ids = tf.argmax(input=sop_logits, axis=2, output_type=tf.int32)
        self._graph_out = {"mlm_pred_ids": mlm_pred_ids, 'mlm_logits': mlm_logits, "sop_pred_ids": sop_pred_ids, 'sop_logits': sop_logits,'masked_index':inputs["masked_index"]}
        return self._graph_out

if __name__ == '__main__':
    final_output=tf.cast([[1.25,2.25,0.5],[1.3,0.4,3.3]],tf.float32)
    pred_ids = tf.argmax(input=final_output, axis=1, output_type=tf.int32)
    one_hot=tf.one_hot(pred_ids,3)
    result=one_hot*final_output
    erg=1-final_output
    print(pred_ids.numpy(),one_hot.numpy(),result.numpy(),erg.numpy())
    logits_so=tf.cast([[[0.1,0.9],[0.2,0.8],[0.3,0.7]],[[0.4,0.6],[0.5,0.5],[0.6,0.4]]],tf.float32)
    print(tf.reduce_mean(tf.transpose(logits_so,[0,2,1]),axis=-1))
    print(tf.cast(tf.round(tf.reduce_mean(tf.cast([[0,1,1,0],[1,1,1,0]],tf.float32),axis=-1)),tf.int32))


