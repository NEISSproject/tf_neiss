import os

from trainer.lav_base import LavBase
import model_fn.model_fn_2d.model_fn_2dtriangle as model_fn_classes
import util.flags as flags
from input_fn.input_fn_2d.input_fn_generator_triangle2d import InputFn2DT

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # set tf log_level to warning(2), default: info(1)
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'  # no tune necessary, short running time

flags.define_string('model_type', 'ModelTriangle', 'Model Type to use choose from: ModelTriangle, ...')
flags.define_string('graph', 'GraphBase', 'GraphBase should be enough to load saved model')
flags.define_boolean('complex_phi', False, "if set: a=phi.real, b=phi.imag, instead of a=cos(phi) b=sin(phi)-1"
                                           "additional flag need for specific input_fn, model, graph")
flags.define_string('loss_mode', 'point3', 'switch loss calculation, see model_fn_2dtriangle.py')
flags.define_boolean('plot', False, "plot results in pdf file, (slow)")
flags.FLAGS.parse_flags()


class LavTriangle2D(LavBase):
    def __init__(self):
        super(LavTriangle2D, self).__init__()
        self._input_fn_generator = InputFn2DT(self._flags)
        self._model = getattr(model_fn_classes, self._flags.model_type)(self._params)


if __name__ == "__main__":
    lav = LavTriangle2D()
    lav.lav()
