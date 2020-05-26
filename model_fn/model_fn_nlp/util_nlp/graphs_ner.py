import tensorflow as tf

from model_fn.graph_base import GraphBase
import model_fn.model_fn_nlp.util_nlp.transformer as bert_graphs
import model_fn.util_model_fn.optimizer as optimizers

class NERwithMiniBERT(GraphBase):
    def set_optimizer(self,params):
        """set a custom optimizer using --optimizer, --optimizer_params,
        - overwrite if you need something different; set self.optimizer = tf.keras.optimizer._class_object"""

        get_optimizer = getattr(optimizers, params['flags'].optimizer)
        self.custom_optimizer = get_optimizer(params)
        self.bert_optimizer = self.custom_optimizer.get_keras_optimizer()
        #self.custom_optimizer.print_params()

    def __init__(self, params):
        super(NERwithMiniBERT, self).__init__(params)
        self._flags = params['flags']
        self._target_vocab_size = params['num_tags'] + 2
        self._pretrained_bert = getattr(bert_graphs, params['flags'].bert_graph)(params)
        self.set_optimizer(params)
        #self._pretrained_bert.load_weights(tf.train.latest_checkpoint(self._flags.bert_dir))
        checkpoint_obj = tf.train.Checkpoint(step=self._pretrained_bert.global_step, optimizer=self.bert_optimizer,
                                             model=self._pretrained_bert)

        if tf.train.get_checkpoint_state(self._flags.bert_dir):
            print("restore bert_model from bert checkpoint: {}".format(self._flags.bert_dir))
            checkpoint_obj.restore(tf.train.latest_checkpoint(self._flags.bert_dir)).expect_partial()

        self._tracked_layers["last_layer"] = tf.keras.layers.Dense(self._target_vocab_size, activation=None,
                                                                   name="last_layer")
        self._tracked_layers["softmax"] = tf.keras.layers.Softmax()

    def call(self, inputs, training=None, mask=None):
        inp={}
        inp["text"]=inputs["sentence"]
        bert_graph_out=self._pretrained_bert(inp)

        final_output = self._tracked_layers["last_layer"](bert_graph_out["enc_output"])  # (batch_size, tar_seq_len, target_vocab_size)
        pred_ids = tf.argmax(input=final_output, axis=2, output_type=tf.int32)
        probabilities = self._tracked_layers["softmax"](final_output)
        self._graph_out = {"pred_ids": pred_ids, 'probabilities': probabilities, 'logits': final_output}
        return self._graph_out
