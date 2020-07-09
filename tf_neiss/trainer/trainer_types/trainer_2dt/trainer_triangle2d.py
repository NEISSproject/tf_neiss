import os
import logging
from trainer.trainer_base import TrainerBase
import model_fn.model_fn_2d.model_fn_2dtriangle as models
import util.flags as flags
from input_fn.input_fn_2d.input_fn_generator_triangle2d import InputFn2DT

# Model parameter
# ===============
flags.define_string('model_type', 'ModelTriangle', 'Model Type to use choose from: ModelTriangle')
flags.define_string('graph', 'GraphMultiFF', 'class name of graph architecture')
flags.define_dict('graph_params', {}, "key=value pairs defining the configuration of the inference class. see used "
                                      "'inference'/'encoder'/'decoder'-class for available options. e.g.["
                                      "mvn (bool), nhidden_lstm_1 (int), nhidden_lstm_2 (int),"
                                      "nhidden_lstm_3 (int), dropout_lstm_1 (float), dropout_lstm_2 (float), "
                                      "dropout_lstm_3 (float), relu_clip (float)]")
flags.define_string('loss_mode', 'point_diff', 'switch loss calculation, see model_fn_2dtriangle.py')
flags.define_integer('data_len', 3142, 'F(phi) amount of values saved in one line')
flags.define_boolean('complex_phi', False, "if set: a=phi.real, b=phi.imag, instead of a=cos(phi) b=sin(phi)-1")
flags.FLAGS.parse_flags()


class Trainer2DTriangle(TrainerBase):
    def __init__(self):
        super(Trainer2DTriangle, self).__init__()
        self._input_fn_generator = InputFn2DT(self._flags)
        self._model_class = getattr(models, self._flags.model_type)
        # self._graph.info()


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    trainer = Trainer2DTriangle()
    trainer.train()
