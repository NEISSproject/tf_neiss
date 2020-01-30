import tensorflow as tf
import numpy as np

from model_fn.graph_base import GraphBase
from model_fn.model_fn_nlp.util_nlp.attention import Selfattention, MultiHeadAttention
from model_fn.model_fn_nlp.util_nlp.transformer import EncoderLayer, Encoder, Decoder


def create_padding_mask(seq):
    seq_mask = tf.cast(tf.sequence_mask(seq), tf.int32)
    masked = tf.cast(tf.math.equal(seq_mask, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return masked[:, tf.newaxis, tf.newaxis, :]


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class KerasGraphFF3(GraphBase):
    def __init__(self, params):
        super(KerasGraphFF3, self).__init__(params)
        self._flags = params['flags']
        self.vocab_tag_size = params['num_tags']
        # declare graph_params and update from dict --graph_params
        self.graph_params["ff_hidden_1"] = 128
        self.graph_params["ff_hidden_2"] = 128
        self.graph_params["ff_hidden_3"] = 128
        # initilize keras layer
        self._tracked_layers["ff_layer_1"] = tf.keras.layers.Dense(self.graph_params["ff_hidden_1"],
                                                                   activation=tf.nn.leaky_relu, name="ff_layer_1")
        self._tracked_layers["ff_layer_2"] = tf.keras.layers.Dense(self.graph_params["ff_hidden_2"],
                                                                   activation=tf.nn.leaky_relu, name="ff_layer_2")
        self._tracked_layers["ff_layer_3"] = tf.keras.layers.Dense(self.graph_params["ff_hidden_3"],
                                                                   activation=tf.nn.leaky_relu, name="ff_layer_3")
        self._tracked_layers["last_layer"] = tf.keras.layers.Dense(self.vocab_tag_size, activation=None,
                                                                   name="last_layer")
        self._tracked_layers["softmax"] = tf.keras.layers.Softmax()

    @tf.function
    def call(self, inputs, training=None, mask=None):
        sentence = inputs["sentence"]
        sentencelength = inputs["sentencelength"]
        # connect keras layers
        ff_layer_1_out = self._tracked_layers["ff_layer_1"](sentence)
        ff_layer_2_out = self._tracked_layers["ff_layer_2"](ff_layer_1_out)
        ff_layer_3_out = self._tracked_layers["ff_layer_3"](ff_layer_2_out)
        logits = self._tracked_layers["last_layer"](ff_layer_3_out)
        pred_ids = tf.argmax(input=logits, axis=2, output_type=tf.int32)
        probabilities = self._tracked_layers["softmax"](logits)
        self._graph_out = {"pred_ids": pred_ids, 'probabilities': probabilities, 'logits': logits,
                           "sentencelength": sentencelength}
        return self._graph_out


class SelfAtt(GraphBase):
    def __init__(self, params):
        super(SelfAtt, self).__init__(params)
        self._flags = params['flags']
        self.vocab_tag_size = params['num_tags']
        # declare graph_params and update from dict --graph_params
        self.graph_params["self_att_num_dims"] = 128
        # initilize keras layer
        self._tracked_layers["self_attention"] = Selfattention(params, 300, self.graph_params["self_att_num_dims"], 0.5)
        self._tracked_layers["last_layer"] = tf.keras.layers.Dense(self.vocab_tag_size, activation=None,
                                                                   name="last_layer")
        self._tracked_layers["softmax"] = tf.keras.layers.Softmax()

    @tf.function
    def call(self, inputs, training=None, mask=None):
        sentence = inputs["sentence"]
        sentencelength = inputs["sentencelength"]
        # connect keras layers
        self_att_out = self._tracked_layers["self_attention"](inputs=sentence, training=training)
        logits = self._tracked_layers["last_layer"](self_att_out)
        pred_ids = tf.argmax(input=logits, axis=2, output_type=tf.int32)
        probabilities = self._tracked_layers["softmax"](logits)
        self._graph_out = {"pred_ids": pred_ids, 'probabilities': probabilities, 'logits': logits,
                           "sentencelength": sentencelength}
        return self._graph_out


class MultiheadAtt(GraphBase):
    def __init__(self, params):
        super(MultiheadAtt, self).__init__(params)
        self._flags = params['flags']
        self.vocab_tag_size = params['num_tags']
        self.embed_dim = 300
        self.num_heads = 10
        self.max_seq_len = 200
        self.pos_encoding = positional_encoding(self.max_seq_len,
                                                self.embed_dim)

        # initilize keras layer
        self._tracked_layers["multihead_attention"] = MultiHeadAttention(self.embed_dim, self.num_heads)
        self._tracked_layers["last_layer"] = tf.keras.layers.Dense(self.vocab_tag_size, activation=None,
                                                                   name="last_layer")
        self._tracked_layers["softmax"] = tf.keras.layers.Softmax()

    def call(self, inputs, training=None, mask=None):
        sentence = inputs["sentence"]
        sentencelength = inputs["sentencelength"]
        sentencelength = sentencelength[:, 0]
        # add pos encoding
        # max_batch_seq_len = tf.shape(sentence)[1]
        # sentence += self.pos_encoding[:, :max_batch_seq_len, :]

        # connect keras layers
        mask = create_padding_mask(sentencelength)
        multihead_att_out, attention_weights = self._tracked_layers["multihead_attention"](
            {'q': sentence, 'k': sentence, 'v': sentence, 'mask': mask})
        logits = self._tracked_layers["last_layer"](multihead_att_out)
        pred_ids = tf.argmax(input=logits, axis=2, output_type=tf.int32)
        probabilities = self._tracked_layers["softmax"](logits)
        self._graph_out = {"pred_ids": pred_ids, 'probabilities': probabilities, 'logits': logits,
                           "sentencelength": sentencelength}
        return self._graph_out


class EncoderLayerAlone(GraphBase):
    def __init__(self, params):
        super(EncoderLayerAlone, self).__init__(params)
        self._flags = params['flags']
        self.vocab_tag_size = params['num_tags']
        self.embed_dim = 300
        self.num_heads = 10
        self.max_seq_len = 200
        self.pos_encoding = positional_encoding(self.max_seq_len,
                                                self.embed_dim)

        # initilize keras layer
        self._tracked_layers["encoder_layer"] = EncoderLayer(self.embed_dim, self.num_heads, self.embed_dim)
        self._tracked_layers["last_layer"] = tf.keras.layers.Dense(self.vocab_tag_size, activation=None,
                                                                   name="last_layer")
        self._tracked_layers["softmax"] = tf.keras.layers.Softmax()

    def call(self, inputs, training=None, mask=None):
        sentence = inputs["sentence"]
        sentencelength = inputs["sentencelength"]
        sentencelength = sentencelength[:, 0]
        # add pos encoding
        # max_batch_seq_len = tf.shape(sentence)[1]
        # sentence += self.pos_encoding[:, :max_batch_seq_len, :]

        # connect keras layers
        mask = create_padding_mask(sentencelength)
        enc_lay_out = self._tracked_layers["encoder_layer"]({'x': sentence, 'mask': mask}, training)
        logits = self._tracked_layers["last_layer"](enc_lay_out)
        pred_ids = tf.argmax(input=logits, axis=2, output_type=tf.int32)
        probabilities = self._tracked_layers["softmax"](logits)
        self._graph_out = {"pred_ids": pred_ids, 'probabilities': probabilities, 'logits': logits,
                           "sentencelength": sentencelength}
        return self._graph_out


class EncoderFull(GraphBase):
    def __init__(self, params):
        super(EncoderFull, self).__init__(params)
        self._flags = params['flags']
        self.vocab_tag_size = params['num_tags']
        self.embed_dim = 300
        self.num_heads = 10
        self.max_seq_len = 200
        self.num_layers = 8
        self.max_pos_encoding = 200
        self.input_vocab_size = 0
        self.pos_encoding = positional_encoding(self.max_seq_len,
                                                self.embed_dim)

        # initilize keras layer
        self._tracked_layers["encoder"] = Encoder(self.num_layers, self.embed_dim, self.num_heads, self.embed_dim,
                                                  self.input_vocab_size, self.max_pos_encoding)
        self._tracked_layers["last_layer"] = tf.keras.layers.Dense(self.vocab_tag_size, activation=None,
                                                                   name="last_layer")
        self._tracked_layers["softmax"] = tf.keras.layers.Softmax()

    def call(self, inputs, training=None, mask=None):
        sentence = inputs["sentence"]
        sentencelength = inputs["sentencelength"]
        sentencelength = sentencelength[:, 0]
        # add pos encoding
        # max_batch_seq_len = tf.shape(sentence)[1]
        # sentence += self.pos_encoding[:, :max_batch_seq_len, :]

        # connect keras layers
        mask = create_padding_mask(sentencelength)
        enc_lay_out = self._tracked_layers["encoder"]({'x': sentence, 'mask': mask}, training)
        logits = self._tracked_layers["last_layer"](enc_lay_out)
        pred_ids = tf.argmax(input=logits, axis=2, output_type=tf.int32)
        probabilities = self._tracked_layers["softmax"](logits)
        self._graph_out = {"pred_ids": pred_ids, 'probabilities': probabilities, 'logits': logits,
                           "sentencelength": sentencelength}
        return self._graph_out


class Transformer(GraphBase):
    def __init__(self, params):
        super(Transformer, self).__init__(params)
        self._flags = params['flags']
        self._num_layers = 1
        self._d_model = 300
        self._num_heads = 10
        self._dff = 300
        self._input_vocab_size = params['tok_size'] + 2
        self._target_vocab_size = params['num_tags'] + 2
        self._pe_input = 300
        self._pe_target = 300
        self._rate = 0.1

        self._tracked_layers["encoder"] = Encoder(self._num_layers, self._d_model, self._num_heads, self._dff,
                                                  self._input_vocab_size, self._pe_input, self._rate)

        self._tracked_layers["decoder"] = Decoder(self._num_layers, self._d_model, self._num_heads, self._dff,
                                                  self._target_vocab_size, self._pe_target, self._rate)

        self._tracked_layers["last_layer"] = tf.keras.layers.Dense(self._target_vocab_size)
        self._tracked_layers["softmax"] = tf.keras.layers.Softmax()

    def call(self, inputs, training=None, mask=None):
        inp = inputs["sentence"]
        sentencelength = inputs["sentencelength"]
        sentencelength = sentencelength[:, 0]
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
        self._graph_out = {"pred_ids": pred_ids, 'probabilities': probabilities, 'logits': final_output,
                           "sentencelength": sentencelength, "attention_weights": attention_weights}
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
