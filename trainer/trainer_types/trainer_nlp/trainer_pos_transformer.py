from trainer.trainer_base import TrainerBase
import model_fn.model_fn_nlp.model_fn_pos_transformer as models
import util.flags as flags
from input_fn.input_fn_nlp.input_fn_pos_transformer import InputFnTFPOS

# Model parameter
# ===============
flags.define_string('model_type', 'ModelTFPOS', 'Model Type to use choose from: ModelTriangle')
flags.define_string('tags', 'stts_tiger.txt', 'path to tag vocabulary')
flags.define_string('tokenizer', '../../../data/tokenizer/tigertokenizer_de', 'path to tokenizer for encoder')
flags.define_string('bert_dir', '../../../bertmodels/bertdeminiwu', 'path to pretrained bert_model checkpoint')
flags.define_string('bert_graph', 'BERTMiniLM', 'class name of pretrained bert graph architecture')
flags.define_string('graph', 'KerasGraphFF3', 'class name of graph architecture')
flags.define_list('add_types', str, 'types that are add features int or float',
                  "", [])
flags.define_integer('buffer', 1,
                     'number of training samples hold in the cache. (effects shuffling)')
flags.define_boolean('predict_mode', False, 'If and only if true the prediction will be accomplished')
flags.FLAGS.parse_flags()


class TrainerTFPOS(TrainerBase):
    def __init__(self):
        super(TrainerTFPOS, self).__init__()
        self._input_fn_generator = InputFnTFPOS(self._flags)
        self._input_fn_generator.print_params()
        self._params['num_tags'] = self._input_fn_generator.get_num_tags()
        if "BERT" in self._flags.graph:
            self._params['tok_size'] = self._input_fn_generator._tok_vocab_size+3
        else:
            self._params['tok_size'] = self._input_fn_generator._tok_vocab_size
        self._params['tags'] = self._input_fn_generator.getTagMapper()
        self._model_class = getattr(models, self._flags.model_type)
        # self._graph.info()


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    trainer = TrainerTFPOS()
    trainer.train()
