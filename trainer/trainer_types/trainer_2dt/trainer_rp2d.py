import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import util.flags as flags
from trainer.trainer_base import TrainerBase
from input_fn.input_fn_2d.input_fn_generator_rp2d import InputFnRegularPolygon2D
import model_fn.model_fn_2d.model_fn_rp2d as models

# Model parameter
# ===============
flags.define_string('model_type', 'ModelRegularPolygon', 'Model Type to use choose from: ModelTriangle')
flags.define_string('graph', 'GraphConv2MultiFF', 'class name of graph architecture')

flags.define_dict('graph_params', {},
                  "key=value pairs defining the configuration of the inference class. see used "
                  "'inference'/'encoder'/'decoder'-class for available options. e.g.["
                  "mvn (bool), nhidden_lstm_1 (int), nhidden_lstm_2 (int),"
                  "nhidden_lstm_3 (int), dropout_lstm_1 (float), dropout_lstm_2 (float), "
                  "dropout_lstm_3 (float), relu_clip (float)]")
flags.define_integer('data_len', 3142, 'F(phi) amount of values saved in one line')
flags.define_integer('max_edges', 8, "Max number of edges must be known (depends on dataset), "
                                     "if unknown pick one which is definitv higher than edges in dataset")
flags.FLAGS.parse_flags()


class TrainerRegularPolygon2D(TrainerBase):
    def __init__(self):
        super(TrainerRegularPolygon2D, self).__init__()
        self._input_fn_generator = InputFnRegularPolygon2D(self._flags)
        self._model_fn = getattr(models, self._flags.model_type)(self._params)
        self._model_fn.info()


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    trainer = TrainerRegularPolygon2D()
    trainer.train()
