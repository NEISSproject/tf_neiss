import tensorflow as tf

from model_fn.graph_base import GraphBase
from model_fn.model_fn_nlp.util_nlp.transformer import Encoder

class BERTMiniSOP(GraphBase):
    def __init__(self, params):
        super(BERTMiniSOP, self).__init__(params)
        self._flags = params['flags']
        self._num_layers = 6
        self._d_model = 512
        self._num_heads = 8
        self._dff = 512
        self._vocab_size = params['tok_size']
        self._pos_enc_max = 20000
        self._rate = 0.1

        self._tracked_layers["encoder"] = Encoder(self._num_layers, self._d_model, self._num_heads, self._dff,
                                                  self._vocab_size, self._pos_enc_max, self._rate)

        self._tracked_layers["last_layer"] = tf.keras.layers.Dense(2)
        #self._tracked_layers["softmax"] = tf.keras.layers.Softmax()

    def call(self, inputs, training=None, mask=None):
        inp = inputs["text"]
        textlength = inputs["textlength"]
        textlength = textlength[:, 0]



        enc_padding_mask = self.create_padding_mask_trans(inp)

        enc_output = self._tracked_layers["encoder"]({'x': inp, 'mask': enc_padding_mask},
                                                     training)  # (batch_size, inp_seq_len, d_model)


        final_output = self._tracked_layers["last_layer"](enc_output)  # (batch_size, tar_seq_len, target_vocab_size)
        pred_ids = tf.argmax(input=final_output, axis=2, output_type=tf.int32)
        #probabilities = self._tracked_layers["softmax"](final_output)
        self._graph_out = {"pred_ids": pred_ids, 'logits': final_output, 'enc_output': enc_output, 'textlength':textlength}
        return self._graph_out


    def create_padding_mask_trans(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]
