from trainer.trainer_base import TrainerBase
import model_fn.model_fn_nlp.model_fn_bert_lm as models
import util.flags as flags
from input_fn.input_fn_nlp.input_fn_bert_lm import InputFnBertLM

# Model parameter
# ===============
flags.define_string('model_type', 'ModelBertLM', 'Model Type to use choose from: ModelTriangle')
flags.define_string('tokenizer', "../../../data/tokenizer/tokenizer_de", 'path to subword tokenizer')
flags.define_string('graph', 'BERTMiniLM', 'class name of graph architecture')
flags.define_list('add_types', str, 'types that are add features int or float',
                  "", [])
flags.define_integer('buffer', 1,
                     'number of training samples hold in the cache. (effects shuffling)')
flags.define_boolean('predict_mode', False, 'If and only if true the prediction will be accomplished')
flags.FLAGS.parse_flags()


class TrainerTFBertLM(TrainerBase):
    def __init__(self):
        super(TrainerTFBertLM, self).__init__()
        self._input_fn_generator = InputFnBertLM(self._flags)
        self._input_fn_generator.print_params()
        self._params['tok_size'] = self._input_fn_generator._tok_vocab_size+3
        self._model_class = getattr(models, self._flags.model_type)
        # self._graph.info()


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    trainer = TrainerTFBertLM()
    trainer.train()
