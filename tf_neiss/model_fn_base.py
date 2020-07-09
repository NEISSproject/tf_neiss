import copy
import logging
import os

import tensorflow as tf
import tensorflow_addons as tfa

import model_fn.util_model_fn.optimizer as optimizers
import util.flags as flags

logger = logging.getLogger(__name__)
# logger.setLevel("DEBUG")


class ModelBase(object):
    def __init__(self, params):
        self._params = copy.deepcopy(params)
        self._params["flags"] = flags.FLAGS
        self._flags = flags.FLAGS
        self._mode_training = True
        self._mode = "train"
        self._net_id = None

        self.graph_train = None
        self.graph_eval = None
        self.summary_writer = dict()
        self.metrics = {"eval": {}, "train": {}}
        if self._flags.hasKey("tensorboard") and self._flags.tensorboard:
            self.summary_writer["train"] = tf.summary.create_file_writer(
                os.path.join(self._flags.checkpoint_dir, "logs", "train"))
            self.summary_writer["eval"] = tf.summary.create_file_writer(
                os.path.join(self._flags.checkpoint_dir, "logs", "eval"))
            self.metrics["eval"]["loss"] = tf.keras.metrics.Mean('loss', dtype=tf.float32)
            self.metrics["train"]["loss"] = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        else:
            self.summary_writer["train"] = None
            self.summary_writer["eval"] = None

        self.target_dict = None
        self._graph_out = None

        # the custom_optimizer includes the learning_rate_schedule, see util_model_fn/optimizer.py
        self.custom_optimizer = None
        # tf.keras.optimzer instance used for training
        self.optimizer = None

    def set_mode(self, mode="train"):
        """switch model between train and eval, assert no-mixed-mode"""
        assert type(self._mode_training) is bool
        assert mode == "train" or mode == "eval"
        if self._mode_training:
            assert self._mode == "train"
        else:
            assert self._mode == "eval"
        self._mode = mode
        if self._mode == "train":
            self._mode_training = True
        else:
            self._mode_training = False

    def set_optimizer(self):
        """set a custom optimizer using --optimizer, --optimizer_params,
        - overwrite if you need something different; set self.optimizer = tf.keras.optimizer._class_object"""

        get_optimizer = getattr(optimizers, self._params['flags'].optimizer)
        self.custom_optimizer = get_optimizer(self._params)
        self.optimizer = self.custom_optimizer.get_keras_optimizer()
        self.custom_optimizer.print_params()

    def set_interface(self, val_dataset):
        build_inputs = next(iter(val_dataset))
        build_out = self.graph_train.call(build_inputs[0], training=False)
        self.graph_train._set_inputs(inputs=build_inputs[0], training=False)
        self.outputs = list()
        self.output_names = list()
        for key in sorted(list(self.graph_train._graph_out)):
            self.outputs.append(build_out[key])
            self.output_names.append(key)
        return build_inputs, build_out

    def get_graph(self):
        """
        Model specific definition of the graph
        :return:
        """

    def get_loss(self):
        """
        Model specific calculation of the loss
        :return:
        """

    def get_predictions(self):
        """
        Model specific calculation of the prediction
        :return:
        """
        pass

    def get_metrics(self):
        """
        Model specific calculation of the metrics
        :return:
        """
        pass

    def print_params(self):
        try:
            print("##### {}:".format("OPTIMIZER"))
            self.optimizer.print_params()
        except AttributeError as ex:
            logging.warning("No Optimizer set. Maybe in LAV mode?")
        try:
            print("##### {}:".format("GRAPH"))
            self.get_graph().print_params()
        except AttributeError as ex:
            logging.warning("No can call self.get_graph().print_params(). Please Debug it!")

    def get_placeholder(self):
        """
        Model specific inputs as placeholder (dict)
        e.g.:
                return {"img": tf.compat.v1.placeholder(tf.float32, [None, self._flags.image_height, None], name="inImg"),
                        "imgLen": tf.compat.v1.placeholder(tf.int32, [None], name="inSeqLen")}
        :return:
        """
        pass

    def serving_input_receiver_fn(self):
        """
        Similar to get_placeholder but puts it into a estimator expected form
        :return:
        """
        inputs = self.get_placeholder()
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    def get_output_nodes(self, has_graph=True):
        """
        Model specific output names
        e.g.:
        if has_graph:
            logits3d = tf.transpose(self._graph_out['logits'], [1, 0, 2])
            tf.identity(logits3d, name="outConfMat")  # name to grab from java
            tf.identity(self._graph_out['outLen'], name="outSeqLen")  # name to grab from java
        return "outConfMat" + "," + "outSeqLen"  # return names as comma separated string without spaces
        :return:
        """
        pass

    def write_tensorboard(self):
        """Write metrics to tensorboard-file (it's called after each epoch) and reset tf.keras.metrics"""
        with self.summary_writer[self._mode].as_default():
            if self._mode_training:
                tf.summary.scalar("learning_rate",
                                  self.custom_optimizer.get_current_learning_rate(self.optimizer.iterations - 1),
                                  step=self.optimizer.iterations - 1)
                tf.summary.scalar("learning_rate", self.custom_optimizer.get_current_learning_rate(self.optimizer.iterations),
                                  step=self.optimizer.iterations)
            else:
                # add
                pass

            logger.info("Reset all metics {}".format(self._mode))
            for metric in self.metrics[self._mode]:
                logger.debug("Write metric: {} with tf.name: {}".format(metric, self.metrics[self._mode][metric].name))
                tf.summary.scalar(metric, self.metrics[self._mode][metric].result(), step=self.graph_train.global_epoch)
                logger.debug("Reset metric: {} with tf.name: {}".format(metric, self.metrics[self._mode][metric].name))
                self.metrics[self._mode][metric].reset_states()

    def to_tensorboard(self, graph_out_dict, targets, input_features):
        """update tf.keras.metrics with this function (it's called after each batch"""
        self.metrics[self._mode]["loss"](graph_out_dict["loss"])

    def export_helper(self):
        """
        Model specific function which is run at the end of export. Purpose e.g. copy preproc etc. to export dir
        :return:
        """
        pass

    def print_evaluate(self, output_nodes_dict, target_dict):
        """is called in lav(load_and_validate) in each batch"""
        pass
        return 1, 1

    def print_evaluate_summary(self):
        """is called at end of lav(load_and_validate), can use graph variables or plot something"""
        pass

    def get_call_graph_signature(self):
        return None
