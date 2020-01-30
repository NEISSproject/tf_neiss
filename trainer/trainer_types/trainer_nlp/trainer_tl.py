from trainer.trainer_base import TrainerBase
import model_fn.model_fn_nlp.model_fn_tl as models
import util.flags as flags
from input_fn.input_fn_nlp.input_fn_tl import InputFnTL

# Model parameter
# ===============
flags.define_string('model_type', 'ModelTL', 'Model Type to use choose from: ModelTriangle')
flags.define_string('tokenizer_inp', "../../../data/tokenizer/tokenizer_pt", 'path to input tokenizer')
flags.define_string('tokenizer_tar', '../../../data/tokenizer/tokenizer_en', 'path to target tokenizer')
flags.define_string('graph', 'KerasGraphFF3', 'class name of graph architecture')
flags.define_list('add_types', str, 'types that are add features int or float',
                  "", [])
flags.define_integer('buffer', 20000,
                     'number of training samples hold in the cache. (effects shuffling)')
flags.define_boolean('predict_mode', False, 'If and only if true the prediction will be accomplished')
flags.FLAGS.parse_flags()


class TrainerTL(TrainerBase):
    def __init__(self):
        super(TrainerTL, self).__init__()
        self._input_fn_generator = InputFnTL(self._flags)
        self._input_fn_generator.print_params()

        self._params['input_vocab_size'] = self._input_fn_generator.get_input_vocab_size()
        self._params['target_vocab_size'] = self._input_fn_generator.get_target_vocab_size()
        self._model_class = getattr(models, self._flags.model_type)
        # self._graph.info()


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    trainer = TrainerTL()
    trainer.train()
