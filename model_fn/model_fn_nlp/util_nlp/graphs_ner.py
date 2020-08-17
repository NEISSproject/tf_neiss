import tensorflow as tf
from model_fn.graph_base import GraphBase
import model_fn.model_fn_nlp.util_nlp.transformer as bert_graphs

class NERwithMiniBERT(GraphBase):
    def __init__(self, params):
        super(NERwithMiniBERT, self).__init__(params)
        self._flags = params['flags']
        self._target_vocab_size = params['num_tags'] + 2
        self._pretrained_bert = getattr(bert_graphs, params['flags'].bert_graph)(params)

        self._tracked_layers["last_layer"] = tf.keras.layers.Dense(self._target_vocab_size, activation=None,
                                                                   name="last_layer")
        self._tracked_layers["softmax"] = tf.keras.layers.Softmax()

    def call(self, inputs, training=None, mask=None):
        inp={}
        inp["text"]=inputs["sentence"]
        bert_graph_out=self._pretrained_bert(inp,training)

        final_output = self._tracked_layers["last_layer"](bert_graph_out["enc_output"])  # (batch_size, tar_seq_len, target_vocab_size)
        pred_ids = tf.argmax(input=final_output, axis=2, output_type=tf.int32)
        probabilities = self._tracked_layers["softmax"](final_output)
        self._graph_out = {"pred_ids": pred_ids, 'probabilities': probabilities, 'logits': final_output}
        return self._graph_out
