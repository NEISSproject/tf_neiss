import copy
import logging
import os

import tensorflow as tf

import model_fn.util_model_fn.optimizer as optimizers
import util.flags as flags


class ModelBase(object):
    def __init__(self, params):
        self._params = copy.deepcopy(params)
        self._params["flags"] = flags.FLAGS
        self._flags = flags.FLAGS

        self._net_id = None

        self.graph_train = None
        self.graph_eval = None

        if self._flags.hasKey("tensorboard") and self._flags.tensorboard:
            self.summary_writer_train = tf.summary.create_file_writer(
                os.path.join(self._flags.checkpoint_dir, "logs", "train"))
            self.summary_writer_eval = tf.summary.create_file_writer(
                os.path.join(self._flags.checkpoint_dir, "logs", "eval"))
            self._train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
            self._eval_loss_metric = tf.keras.metrics.Mean('eval_loss', dtype=tf.float32)
        else:
            self.summary_writer_train = None
            self.summary_writer_eval = None

        self.target_dict = None
        self._graph_out = None

        # the custom_optimizer includes the learning_rate_schedule, see util_model_fn/optimizer.py
        self.custom_optimizer = None
        # tf.keras.optimzer instance used for training
        self.optimizer = None

    def set_optimizer(self):
        """set a custom optimizer using --optimizer, --optimizer_params,
        - overwrite if you need something different; set self.optimizer = tf.keras.optimizer._class_object"""

        get_optimizer = getattr(optimizers, self._params['flags'].optimizer)
        self.custom_optimizer = get_optimizer(self._params)
        self.optimizer = self.custom_optimizer.get_keras_optimizer()
        self.custom_optimizer.print_params()

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
        with self.summary_writer_train.as_default():
            tf.summary.scalar("train_loss", self._train_loss_metric.result(), step=self.graph_train.global_epoch)
        with self.summary_writer_train.as_default():
            tf.summary.scalar("learing_rate",
                              self.custom_optimizer.get_current_learning_rate(self.optimizer.iterations - 1),
                              step=self.optimizer.iterations - 1)
            tf.summary.scalar("learing_rate", self.custom_optimizer.get_current_learning_rate(),
                              step=self.optimizer.iterations)
            # tf.summary.scalar("learning_rate", self.optimizer.)
        with self.summary_writer_eval.as_default():
            tf.summary.scalar("eval_loss", self._eval_loss_metric.result(), step=self.graph_eval.global_epoch)
        self._eval_loss_metric.reset_states()
        self._train_loss_metric.reset_states()

    def to_tensorboard_train(self, graph_out_dict, targets, input_features):
        """update tf.keras.metrics with this function (it's called after each train-batch"""
        self._train_loss_metric(graph_out_dict["loss"])

    def to_tensorboard_eval(self, graph_out_dict, targets, input_features):
        """update tf.keras.metrics with this function (it's called after each train-batch"""
        self._eval_loss_metric(graph_out_dict["loss"])

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
