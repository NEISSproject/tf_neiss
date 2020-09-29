from trainer.trainer_base import TrainerBase
import model_fn.model_fn_nlp.model_fn_bert_nsp as models
import util.flags as flags
from input_fn.input_fn_nlp.input_fn_bert_nsp import InputFnBertNSP
import os
import time

import tensorflow as tf
from util.misc import get_commit_id

# Model parameter
# ===============
flags.define_string('model_type', 'ModelBertNSP', 'Model Type to use choose from: ModelTriangle')
flags.define_string('tokenizer', "../../../data/tokenizer/tokenizer_de", 'path to subword tokenizer')
flags.define_string('graph', 'BERTMiniNSPAS', 'class name of graph architecture')
flags.define_list('add_types', str, 'types that are add features int or float',
                  "", [])
flags.define_integer('buffer', 1,
                     'number of training samples hold in the cache. (effects shuffling)')
flags.define_integer('max_words_text_part', 40,
                     'maximal number of words in a text part of the input function')
flags.define_boolean('predict_mode', False, 'If and only if true the prediction will be accomplished')
flags.define_string('bert_checkpoint_dir', '', 'Checkpoint to save pure bert model information in.')
flags.define_boolean('rel_pos_enc',False, 'If and only if True the bert uses a relative positional encoding instead of an absolute one')
flags.FLAGS.parse_flags()


class TrainerTFBertNSP(TrainerBase):
    def __init__(self):
        super(TrainerTFBertNSP, self).__init__()
        self._input_fn_generator = InputFnBertNSP(self._flags)
        self._input_fn_generator.print_params()
        self._params['tok_size'] = self._input_fn_generator._tok_vocab_size+3
        self._model_class = getattr(models, self._flags.model_type)
        # self._graph.info()


    def save_bert(self):
        bert_checkpoint_obj = tf.train.Checkpoint(step=self._model.graph_train.global_step, optimizer=self._model.optimizer,
                                             model=self._model.graph_train.bert)
        bert_checkpoint_manager = tf.train.CheckpointManager(checkpoint=bert_checkpoint_obj, directory=self._flags.bert_checkpoint_dir,
                                                        max_to_keep=1)
        bert_checkpoint_manager.save()



if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    trainer = TrainerTFBertNSP()
    trainer.train()
    trainer.save_bert()
