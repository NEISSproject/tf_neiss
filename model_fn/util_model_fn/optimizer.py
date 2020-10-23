import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

from util.flags import update_params


class DecayOptimizer(object):
    """BaseOptimzer holding a _keras_optimizer with learnrate schedule"""

    def __init__(self, params):
        super(DecayOptimizer, self).__init__()
        self._name = "DecayOptimizer"
        self._keras_optimizer = None
        self._learn_rate_schedule = None
        self._params = params
        self._flags = params['flags']
        self._optimizer_params = dict()
        # Default params for the decay scenario
        self._optimizer_params["optimizer"] = 'adam'  # learning rate decay, 1.0 means no decay
        self._optimizer_params["learning_rate"] = 0.01  # initial learning rate
        self._optimizer_params["lr_decay_rate"] = 0.99  # learning rate decay, 1.0 means no decay
        self._optimizer_params["calc_ema"] = False  # enable exponential moving average on trainable variables
        self._optimizer_params["ema_decay"] = 0.1  # set decay of moving average variables
        self._optimizer_params[
            "learning_circle"] = 3  # number of epochs with same learning rate, except for restart-strategies
        self._optimizer_params["d_model"] = 128
        self._optimizer_params["warmup_steps"] = 4000
        self.update_params()

    def update_params(self):
        """Updating of the default params if provided via flags as a dict"""
        self._optimizer_params = update_params(self._optimizer_params, self._flags.optimizer_params, "Optimizer")

    def _get_lr(self):
        """return: tf.keras.optimizer.schedules.LearningRateSchedule class-instance"""
        if not self._learn_rate_schedule:
            decay_steps = self._flags.samples_per_epoch // self._flags.train_batch_size * self._optimizer_params[
                "learning_circle"]
            self._learn_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self._optimizer_params["learning_rate"],
                decay_steps=decay_steps, decay_rate=self._optimizer_params["lr_decay_rate"], staircase=True)
        return self._learn_rate_schedule

    def get_current_learning_rate(self, step=None):
        if step:
            return self._learn_rate_schedule(step)
        else:
            return self._learn_rate_schedule(self._keras_optimizer.iterations)

    def get_keras_optimizer(self):
        """return tf.keras.optimizer.Optimizer_V2 class-instance"""
        if not self._keras_optimizer:
            lr = self._get_lr()
            if self._optimizer_params["optimizer"] == 'adam':
                self._keras_optimizer = tf.keras.optimizers.Adam(lr)
            if self._optimizer_params["optimizer"] == 'rmsprop':
                self._keras_optimizer = tf.keras.optimizers.RMSprop(lr)
            if self._optimizer_params["optimizer"] == 'sgd':
                self._keras_optimizer = tf.keras.optimizers.SGD(lr)
            if self._optimizer_params["optimizer"] == 'lamb':
                print('Use LAMB Optimizer')
                self._keras_optimizer = tfa.optimizers.LAMB(lr)
        if self._optimizer_params["calc_ema"]:
            self._keras_optimizer = tfa.optimizers.MovingAverage(self._keras_optimizer,
                                                                 average_decay=self._optimizer_params["ema_decay"])
        return self._keras_optimizer

    def print_params(self):
        print("{}_params for ".format("optimizer") + self._name + ":")
        sorted_dict = sorted(self._optimizer_params.items(), key=lambda kv: kv[0])
        for a in sorted_dict:
            print("  {}: {}".format(a[0], a[1]))


class FinalDecayOptimizer(DecayOptimizer):
    def __init__(self, params):
        super(FinalDecayOptimizer, self).__init__(params)
        self._name = "FinalDecayOptimizer"
        self._params = params
        self._flags = params['flags']
        # Default params for the decay scenario
        self._optimizer_params["final_epochs"] = 50  # number epochs with reducing learning rate
        self._optimizer_params["decay_fraction"] = 0.001  # reduce to this fraction of LR

    def _get_lr(self):
        """override learning rate schedule of the inherited class add cosine decrease in the final epochs"""
        if not self._learn_rate_schedule:
            self._learn_rate_schedule = CosineDecaySchedule(self._optimizer_params["learning_rate"],
                                     self._flags.samples_per_epoch // self._flags.train_batch_size,
                                     self._optimizer_params["lr_decay_rate"],
                                     self._optimizer_params["decay_fraction"],
                                     self._flags.epochs,
                                     self._optimizer_params["final_epochs"])
        return self._learn_rate_schedule

class WarmupDecayOptimizer(DecayOptimizer):
    def __init__(self, params):
        super(WarmupDecayOptimizer, self).__init__(params)
        self._name = "WarmupDecayOptimizer"
        self._params = params
        self._flags = params['flags']

    def _get_lr(self):
        """override learning rate schedule of the inherited class add cosine decrease in the final epochs"""
        if not self._learn_rate_schedule:
            self._learn_rate_schedule = WarmupSchedule(self._optimizer_params["d_model"],self._optimizer_params["warmup_steps"])
        return self._learn_rate_schedule

    def get_current_learning_rate(self, step=None):
        if step:
            return self._learn_rate_schedule(tf.cast(step,tf.float32))
        else:
            return self._learn_rate_schedule(self._keras_optimizer.iterations)


class CosineDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate,
                 steps_per_epoch,
                 decay,
                 alpha,  # decay_fraction
                 epochs,
                 final_epochs,
                 name=None):
        super(CosineDecaySchedule, self).__init__()
        """piecewise definde function:
        from 0 to warmup_epoch: linear increas from learningrate to warmup_factor*learningrate
        from warmup_epoch to epochs - final_epochs: decay using alpha and learning circle
        from epochs - final_epochs to end: cosine cooldown like in adam final/cosine_decay"""
        self._learning_rate = learning_rate
        self._decay = decay
        self._alpha = alpha
        self._steps_per_epoch = steps_per_epoch
        self._final_epochs = final_epochs
        self._epochs = epochs
        self.name = name

    def decay(self):
        return self._learning_rate * (self._decay ** math_ops.floor(self._step / self._steps_per_epoch))

    def cosine_decrease(self):
        epoch = self._step / self._steps_per_epoch
        return self.decay() * (self._alpha + (1 - self._alpha) * (0.5 + 0.5 * math_ops.cos(
            (epoch - self._epochs + self._final_epochs) / self._final_epochs * 3.14159)))

    def __call__(self, step, *args, **kwargs):
        self._step = step
        self._epoch = math_ops.floor(step / self._steps_per_epoch)
        lam = control_flow_ops.cond(
            math_ops.less_equal(self._epoch, self._epochs - self._final_epochs),
            self.decay, self.cosine_decrease)
        return lam

    def get_config(self):
        return {
            "initial_learning_rate": self._learning_rate,
            "decay_steps": self._steps_per_epoch,
            "decay_rate": self._decay,
            "staircase": True,
            "name": self.name
        }

class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(WarmupSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


if __name__ == '__main__':
    #import matplotlib.pyplot as plt
    import pylab
    import numpy as np
    temp_learning_rate_schedule = WarmupSchedule(512,100000)
    comp_temp_learning_rate_schedule = WarmupSchedule(512,375000)
    comp2_temp_learning_rate_schedule = WarmupSchedule(128,4000)
    def get_current_learning_rate(temp_learning_rate_schedule, step=None):
        if step:
            return temp_learning_rate_schedule(tf.cast(step,tf.float32))
        else:
            return temp_learning_rate_schedule(step)
    x=np.linspace(1,1000000,1000)
    #print(x)
    y=[get_current_learning_rate(temp_learning_rate_schedule,x[i]) for i in range(len(x))]
    pylab.plot(x,y)
    #y=[get_current_learning_rate(comp_temp_learning_rate_schedule,x[i]) for i in range(len(x))]
    #pylab.plot(x,y)#,'co')
    #y=[get_current_learning_rate(comp2_temp_learning_rate_schedule,x[i]) for i in range(len(x))]
    #pylab.plot(x,y)
    pylab.show()
