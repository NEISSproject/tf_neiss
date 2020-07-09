import logging
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from itertools import permutations
import tensorflow as tf

import model_fn.model_fn_2d.util_2d.graphs_2d as graphs
from model_fn.model_fn_base import ModelBase


class ModelPolygonClassifier(ModelBase):
    def __init__(self, params):
        super(ModelPolygonClassifier, self).__init__(params)
        self._flags = self._params['flags']
        self._targets = None
        self._point_dist = None
        self._summary_object = {"tgt_points": [], "pre_points": [], "ordered_best": [], "unordered_best": []}

    def get_graph(self):
        return getattr(graphs, self._params['flags'].graph)(self._params)

    def get_placeholder(self):
        return {"fc": tf.compat.v1.placeholder(tf.float32, [None, 3, None], name="infc")}

    def get_output_nodes(self, has_graph=True):
        if has_graph:
            tf.identity(self._graph_out['edges'], name="edges")  # name to grab from java
        return "e_pred"  # return names as comma separated string without spaces

    def get_target_keys(self):
        return 'edges'

    def get_predictions(self):
        return self._graph_out['e_pred']

    def info(self):
        self.get_graph().print_params()

    def get_loss(self):
        loss = 0.0
        if 'softmax_crossentropy' == self._flags.loss_mode:
            loss += tf.sqrt(tf.compat.v1.losses.softmax_cross_entropy(
                tf.one_hot(tf.squeeze(self._targets['edges'], axis=-1) - 3, depth=4), self._graph_out['e_pred']))
        elif "abs_diff" == self._flags.loss_mode:
            loss += tf.compat.v1.losses.absolute_difference(self._targets['edges'], self._graph_out['e_pred'])
        else:
            logging.error("no valid loss-mode in loss_params")
            raise AttributeError
        loss = tf.reduce_mean(loss)
        return loss

    def export_helper(self):
        for train_list in self._params['flags'].train_lists:
            data_id = os.path.basename(train_list)[:-8]
            shutil.copy(os.path.join("data/synthetic_data", data_id, "log_{}_train.txt".format(data_id)),
                        os.path.join(self._params['flags'].checkpoint_dir, "export"))
        data_id = os.path.basename(self._params['flags'].val_list)[:-8]
        shutil.copy(os.path.join("data/synthetic_data", data_id, "log_{}_val.txt".format(data_id)),
                    os.path.join(self._params['flags'].checkpoint_dir, "export"))

    def print_evaluate(self, output_dict, target_dict):
        with tf.compat.v1.Session().as_default():
            step_diff = target_dict["edges"] - output_dict["e_pred"]
            step_diff_np = step_diff.eval()
            step_diff_listpart = list(step_diff_np)
            self._summary_object["step_diff_list"].extend(step_diff_listpart)

        return 1, 1

    def print_evaluate_summary(self):
        step_diff_arr = np.array(self._summary_object["step_diff_list"])
        fig = plt.figure()
        print("run eval summary")
        pre_points = plt.hist(step_diff_arr, bins=np.max(step_diff_arr) - np.min(step_diff_arr))

        def get_current_epoch_from_file():

            if os.path.isfile(os.path.join(self._flags.checkpoint_dir, "current_epoch.info")):
                with open(os.path.join(self._flags.checkpoint_dir, "current_epoch.info"), "r") as f:
                    current_epoch = int(f.read())
            else:
                current_epoch = int(-1)
            return current_epoch

        epoch = get_current_epoch_from_file()
        pdf = os.path.join(self._params['flags'].model_dir, "error_hist_epoch_{}.pdf".format(epoch))
        fig.savefig(pdf)
        plt.clf()
        zero_ammount = step_diff_arr.shape[0] - np.count_nonzero(step_diff_arr)

        print("full-correct accuracy: {}%".format(zero_ammount / step_diff_arr.shape[0] * 100))
