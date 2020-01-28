from trainer.trainer_base import TrainerBase
import model_fn.model_fn_nlp.model_fn_pos as models
import util.flags as flags
from input_fn.input_fn_nlp.input_fn_pos import InputFnPOS

# Model parameter
# ===============
flags.define_string('model_type', 'ModelPOS', 'Model Type to use choose from: ModelTriangle')
flags.define_string('tags', 'stts_tiger.txt', 'path to tag vocabulary')
flags.define_string('word_embeddings', '../../../data/word_embeddings/cc.de.300.bin', 'path to word embeddings')
flags.define_string('graph', 'KerasGraphFF3', 'class name of graph architecture')
flags.define_list('add_types', str, 'types that are add features int or float',
                  "", [])
flags.define_integer('buffer', 1,
                     'number of training samples hold in the cache. (effects shuffling)')
flags.define_boolean('predict_mode', False, 'If and only if true the prediction will be accomplished')
flags.FLAGS.parse_flags()


class TrainerPOS(TrainerBase):
    def __init__(self):
        super(TrainerPOS, self).__init__()
        self._input_fn_generator = InputFnPOS(self._flags)
        self._input_fn_generator.print_params()

        self._params['num_tags'] = self._input_fn_generator.get_num_tags()
        self._params['tags'] = self._input_fn_generator.getTagMapper()
        self._model_class = getattr(models, self._flags.model_type)
        # self._graph.info()


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    trainer = TrainerPOS()
    trainer.train()
