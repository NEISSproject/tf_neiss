import os
import time

import tensorflow as tf

import tf_neiss.flags as flags
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags.define_string('model_dir', '', 'dir with "export"-folder which was checkpoint_dir before or path to "export"')
# flags.define_string('val_list', 'lists/dummy_val.lst', '.lst-file specifying the dataset used for validation')
# flags.define_integer('val_batch_size', 100, 'number of elements in a val_batch')
# flags.define_list('gpu_devices', int, 'space seperated list of GPU indices to use. ', " ", [])
flags.FLAGS.parse_flags()
flags.define_float('gpu_memory', 0.0, 'amount of gpu memory in MB if set above 0')
flags.define_string("debug_dir", "", "specify dir to save debug outputs, saves are model specific ")
flags.define_integer("batch_limiter", -1, "set to positiv value to stop validation after this number of batches")
flags.FLAGS.parse_flags()


class LavBase(object):
    def __init__(self):
        self._flags = flags.FLAGS
        flags.print_flags()
        self.set_run_config()
        self._input_fn_generator = None
        self._val_dataset = None
        self._graph_eval = None
        self._model = None
        self._model_fn_classes = None
        self._params = None
        self._params = {'num_gpus': len(self._flags.gpu_devices)}
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    def lav(self):
        print("run Load and Validate...")
        val_loss = 0.0
        t1_val = time.time()
        if os.path.isdir(os.path.join(flags.FLAGS.model_dir, "export")):
            export_dir_path = os.path.join(flags.FLAGS.model_dir, "export")
        else:
            export_dir_path = flags.FLAGS.model_dir
        self._model.graph_eval = tf.keras.models.load_model(export_dir_path)
        for (batch, (input_features, targets)) in enumerate(self._input_fn_generator.get_input_fn_val()):
            if self._flags.batch_limiter != -1 and self._flags.batch_limiter <= batch:
                print(
                    "stop validation after {} batches with {} samples each.".format(batch, self._flags.val_batch_size))
                break
            self._model.graph_eval._graph_out = self._model.graph_eval(input_features, training=False)
            loss = tf.keras.losses.mean_absolute_error(self._model.graph_eval._graph_out["pre_points"],
                                                       tf.reshape(targets["points"], (-1, 6)))
            self._model.print_evaluate(self._model.graph_eval._graph_out, targets)
            val_loss += tf.reduce_mean(loss)
        val_loss /= float(batch + 1.0)
        print(
            "val-loss:{:10.3f}, samples/seconde: {:1.1f}".format(val_loss, (batch + 1) * flags.FLAGS.val_batch_size / (
                    time.time() - t1_val)))
        print("Time: {:8.2f}".format(time.time() - t1_val))
        self._model.print_evaluate_summary()

        print("finished")

    def set_run_config(self):
        if flags.FLAGS.hasKey("force_eager") and flags.FLAGS.force_eager:
            tf.config.experimental_run_functions_eagerly(run_eagerly=True)

        gpu_list = ','.join(str(x) for x in flags.FLAGS.gpu_devices)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        print("GPU-DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
