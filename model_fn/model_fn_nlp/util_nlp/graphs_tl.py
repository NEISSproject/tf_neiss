import tensorflow as tf

from model_fn.graph_base import GraphBase
from model_fn.model_fn_nlp.util_nlp.transformer import  Encoder, Decoder

class Transformer(GraphBase):
    def __init__(self, params):
        super(Transformer, self).__init__(params)
        self._flags = params['flags']
        self._num_layers = 4
        self._d_model = 128
        self._num_heads = 8
        self._dff = 512
        self._input_vocab_size = params['input_vocab_size']
        self._target_vocab_size = params['target_vocab_size']
        self._pe_input = params['input_vocab_size']
        self._pe_target = params['target_vocab_size']
        self._rate = 0.1

        self._tracked_layers["encoder"] = Encoder(self._num_layers, self._d_model, self._num_heads, self._dff,
                                                  self._input_vocab_size, self._pe_input, self._rate)

        self._tracked_layers["decoder"] = Decoder(self._num_layers, self._d_model, self._num_heads, self._dff,
                                                  self._target_vocab_size, self._pe_target, self._rate)

        self._tracked_layers["last_layer"] = tf.keras.layers.Dense(self._target_vocab_size)
        self._tracked_layers["softmax"] = tf.keras.layers.Softmax()

    def call(self, inputs, training=None, mask=None):
        inp = inputs["inputs"]
        tar = inputs["tar_inp"]

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)

        enc_output = self._tracked_layers["encoder"]({'x': inp, 'mask': enc_padding_mask},
                                                     training)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self._tracked_layers["decoder"]({
            'tar': tar, 'enc_output': enc_output, 'look_ahead_mask': look_ahead_mask, 'padding_mask': dec_padding_mask},
            training)

        final_output = self._tracked_layers["last_layer"](dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        pred_ids = tf.argmax(input=final_output, axis=2, output_type=tf.int32)
        probabilities = self._tracked_layers["softmax"](final_output)
        self._graph_out = {"pred_ids": pred_ids, 'probabilities': probabilities, 'logits': final_output, "attention_weights": attention_weights}
        return self._graph_out

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = self.create_padding_mask_trans(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = self.create_padding_mask_trans(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask_trans(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    def create_padding_mask_trans(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]
