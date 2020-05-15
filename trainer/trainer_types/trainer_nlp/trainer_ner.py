from trainer.trainer_base import TrainerBase
import model_fn.model_fn_nlp.model_fn_ner as models
import util.flags as flags
from input_fn.input_fn_nlp.input_fn_ner import InputFnNER

# Model parameter
# ===============
flags.define_string('model_type', 'ModelNER', 'Model Type to use choose from: ModelTriangle')
flags.define_string('tags', 'ner_uja.txt', 'path to tag vocabulary')
flags.define_string('tokenizer', '../../../data/tokenizer/tigertokenizer_de', 'path to tokenizer for encoder')
flags.define_string('bert_dir', '../../../bertmodels/bertdeminiwu', 'path to pretrained bert_model checkpoint')
flags.define_string('bert_graph', 'BERTMini', 'class name of pretrained bert graph architecture')
flags.define_string('graph', 'NERwithMiniBERT', 'class name of graph architecture')
flags.define_list('add_types', str, 'types that are add features int or float',
                  "", [])
flags.define_integer('buffer', 1,
                     'number of training samples hold in the cache. (effects shuffling)')
flags.define_boolean('predict_mode', False, 'If and only if true the prediction will be accomplished')
flags.FLAGS.parse_flags()


class TrainerNER(TrainerBase):
    def __init__(self):
        super(TrainerNER, self).__init__()
        self._input_fn_generator = InputFnNER(self._flags)
        self._input_fn_generator.print_params()
        self._params['num_tags'] = self._input_fn_generator.get_num_tags()
        self._params['tok_size'] = self._input_fn_generator._tok_vocab_size+3
        self._params['tags'] = self._input_fn_generator.getTagMapper()
        self._model_class = getattr(models, self._flags.model_type)
        # self._graph.info()


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    trainer = TrainerNER()
    trainer.train()
