import tensorflow as tf

from model_fn.graph_base import GraphBase
from model_fn.model_fn_nlp.util_nlp.transformer import Encoder,BERTMini

class BERTMiniSOP(GraphBase):
    def __init__(self, params):
        super(BERTMiniSOP, self).__init__(params)
        self._flags = params['flags']
        self.bert=BERTMini(params)

        self._tracked_layers["last_layer"] = tf.keras.layers.Dense(params['tok_size'])
        self._tracked_layers["softmax"] = tf.keras.layers.Softmax()

    def call(self, inputs, training=None, mask=None):
        bert_out = self.bert(inputs)

        final_output = self._tracked_layers["last_layer"](bert_out['enc_output'])  # (batch_size, tar_seq_len, target_vocab_size)
        pred_ids = tf.argmax(input=final_output, axis=2, output_type=tf.int32)
        probabilities = self._tracked_layers["softmax"](final_output)
        self._graph_out = {"pred_ids": pred_ids, 'logits': final_output, 'probs':probabilities,'masked_index':inputs["masked_index"]}
        return self._graph_out

if __name__ == '__main__':
    final_output=tf.cast([[1.25,2.25,0.5],[1.3,0.4,3.3]],tf.float32)
    pred_ids = tf.argmax(input=final_output, axis=1, output_type=tf.int32)
    one_hot=tf.one_hot(pred_ids,3)
    result=one_hot*final_output
    erg=1-final_output
    print(pred_ids.numpy(),one_hot.numpy(),result.numpy(),erg.numpy())


