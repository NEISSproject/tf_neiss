import logging
import os
import shutil

import numpy as np
from itertools import permutations
import tensorflow as tf

import model_fn.model_fn_2d.util_2d.graphs_2d as graphs
from model_fn.model_fn_base import ModelBase
from model_fn.util_model_fn.losses import batch_point3_loss, ordered_point3_loss


class ModelPolygon(ModelBase):
    def __init__(self, params):
        super(ModelPolygon, self).__init__(params)
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
            tf.identity(self._graph_out['e_pred'], name="e_pred")  # name to grab from java
        return "e_pred"  # return names as comma separated string without spaces

    def get_target_keys(self):
        return 'edges'

    def get_predictions(self):
        return self._graph_out['e_pred']

    def info(self):
        self.get_graph().print_params()

    def get_loss(self):
        # self._targets['points'] = tf.Print(self._targets['points'], [self._targets['points']])
        # loss0 = tf.losses.absolute_difference(self._targets['points'], self._graph_out['p_pred'])
        # print("params train batch size", self._params["flags"].train_batch_size)
        # print("points", self._targets['points'])
        loss = 0.0
        loss_edge = 0
        # loss_p = tf.print("tgt:", tf.squeeze(self._targets['edges']-3, axis=-1), "\npred", tf.argmax(self._graph_out['e_pred'], axis=1), tf.shape(self._graph_out['e_pred']), summarize=1000)
        # with tf.control_dependencies([loss_p]):
        if 'softmax_crossentropy' == self._flags.loss_mode:
            loss_edge = tf.reduce_mean(tf.sqrt(tf.compat.v1.losses.softmax_cross_entropy(
                tf.one_hot(tf.squeeze(self._targets['edges'], axis=-1) - 3, depth=4), self._graph_out['e_pred'])))
        elif "abs_diff" == self._flags.loss_mode:
            loss_edge = tf.reduce_mean(
                tf.compat.v1.losses.absolute_difference(self._targets['edges'], self._graph_out['e_pred']))
        else:
            logging.error("no valid loss-mode in loss_params")
            raise AttributeError

        def num_step(x, bias):
            return tf.math.divide(1, tf.add(tf.constant(1.0), tf.math.exp(-10.0 * (x - bias))))

        ff_final_reshaped = tf.reshape(tf.cast(self._graph_out['p_pred'], dtype=tf.float32),
                                       shape=(-1, self._flags.max_edges, 2))
        num_step_tensor = 1.0 - num_step(
            tf.cast(tf.range(self._flags.max_edges), dtype=tf.float32) + 1.0,
            (tf.minimum(tf.maximum(tf.cast(self._targets['edges'], dtype=tf.float32), 0.0),
                        tf.cast(self._flags.max_edges, dtype=tf.float32) - 2.0) + 3.5))
        num_step_tensor = tf.expand_dims(num_step_tensor, axis=-1)
        num_step_tensor_broadcast = tf.broadcast_to(num_step_tensor, [tf.shape(ff_final_reshaped)[0], 6, 2])
        print(num_step_tensor_broadcast)
        print(self._targets["points"])
        corrected_p_pred = tf.math.multiply(ff_final_reshaped, num_step_tensor_broadcast)
        # paddings = tf.constant([[0, 0], [0,0], [0,tf.shape()]])
        # target_extpanded = tf.pad(self._targets["points"], )
        loss_points = tf.reduce_mean(tf.losses.absolute_difference(self._targets["points"], corrected_p_pred))

        loss_points = tf.Print(loss_points, [self._targets['edges'], loss_edge, loss_points], summarize=1000,
                               message="loss_edge, loss_points")
        loss_points = tf.Print(loss_points, [self._targets["points"]], summarize=1000, message="tgt_points")
        loss_points = tf.Print(loss_points, [corrected_p_pred], summarize=1000, message="pre_points")
        # loss_points = tf.Print(loss_points, [tf.losses.absolute_difference(self._targets["points"], corrected_p_pred)], summarize=1000, message="point_loss")
        # loss0 = tf.cast(batch_point3_loss(self._targets['points'], self._graph_out['p_pred'],
        #                                   self._params["flags"].train_batch_size), dtype=tf.float32)
        # loss0 = tf.cast(ordered_point3_loss(self._targets['points'], self._graph_out['p_pred'],
        #                                   self._params["flags"].train_batch_size), dtype=tf.float32)
        # target_mom = tf.reduce_mean(
        #     tf.nn.moments(tf.reshape(self._targets['points'], (self._params["flags"].train_batch_size, 6)), axes=1),
        #     axis=1)
        # pred_mom = tf.reduce_mean(
        #     tf.nn.moments(tf.reshape(self._graph_out['p_pred'], (self._params["flags"].train_batch_size, 6)), axes=1),
        #     axis=1)
        # loss1 = tf.losses.absolute_difference(tf.reduce_sum(tf.abs(self._targets['points'])), tf.reduce_sum(tf.abs(self._graph_out['p_pred'])))
        # loss = tf.Print(loss, [loss], message="loss:"
        # print(loss0)
        loss = loss_edge + loss_points
        # loss = tf.Print(loss, [loss, tf.shape(target_mom), target_mom, pred_mom], message="loss0, loss1:")
        # loss = tf.Print(loss, [loss], message="loss:")
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
            tgt_area_sum = 0
            area_diff_sum = 0
            loss_ordered = ordered_point3_loss(output_dict["p_pred"], target_dict["points"],
                                               self._params['flags'].val_batch_size)
            loss_best = batch_point3_loss(output_dict["p_pred"], target_dict["points"],
                                          self._params['flags'].val_batch_size)

            self._summary_object["tgt_points"].extend(
                [target_dict["points"][x] for x in range(self._params['flags'].val_batch_size)])
            self._summary_object["pre_points"].extend(
                [output_dict["p_pred"][x] for x in range(self._params['flags'].val_batch_size)])
            ob_buffer_list = []
            ub_buffer_list = []
            for i in range(output_dict["p_pred"].shape[0]):
                # print("## {:4d} Sample ##".format(i))
                # # print(loss_ordered.eval())
                # print("loss: {:3.2f}(ordered)| {:3.2f} (best)".format(loss_ordered.eval()[i], loss_best.eval()[i]))
                if np.abs(loss_ordered.eval()[i] - loss_best.eval()[i]) > 0.01:
                    # print("WARNING: losses are not equal")
                    ob_buffer_list.append(np.nan)
                    ub_buffer_list.append(loss_best.eval()[i])
                else:
                    ob_buffer_list.append(loss_best.eval()[i])
                    ub_buffer_list.append(np.nan)

            self._summary_object["ordered_best"].extend(ob_buffer_list)
            self._summary_object["unordered_best"].extend(ub_buffer_list)

            # print("predicted points")
            # print(output_dict["p_pred"][i])
            # print("target points")
            # print(target_dict["points"][i])
            # pred_area = np.abs(np.dot((output_dict["p_pred"][i][0] - output_dict["p_pred"][i][1]), (output_dict["p_pred"][i][1] - output_dict["p_pred"][i][2])) / 2.0)
            # tgt_area = np.abs(np.dot((target_dict["points"][i][0] - target_dict["points"][i][1]), (target_dict["points"][i][1] - target_dict["points"][i][2])) / 2.0)
            # area_diff_sum += np.max(pred_area - tgt_area)
            # tgt_area_sum += tgt_area
            # print("area diff: {:0.3f}".format(np.abs(pred_area - tgt_area) / tgt_area))
            # print("target area: {:0.3f}".format(np.abs(tgt_area)))

        return area_diff_sum, tgt_area_sum

    # def print_evaluate_summary(self):
    #     sample_counter = 0
    #     import matplotlib.pyplot as plt
    #
    #     from matplotlib.patches import Polygon
    #     from shapely import geometry
    #     from matplotlib.collections import PatchCollection
    #     summary_lenght= len(self._summary_object["tgt_points"])
    #     print("summary length: {}".format(summary_lenght))
    #
    #     tgt_area_arr = np.zeros(summary_lenght)
    #     pre_area_arr = np.zeros(summary_lenght)
    #     pre_area_arr = np.zeros(summary_lenght)
    #     iou_arr = np.zeros(summary_lenght)
    #     co_loss_arr = np.ones(summary_lenght) * np.nan
    #     wo_loss_arr = np.ones(summary_lenght) * np.nan
    #
    #     for i in range(summary_lenght):
    #         pre_points = np.reshape(self._summary_object["pre_points"][i], (3,2))
    #         tgt_points = np.reshape(self._summary_object["tgt_points"][i], (3,2))
    #         # print(pre_points)
    #         # print(tgt_points)
    #         pre_polygon = geometry.Polygon([pre_points[0], pre_points[1], pre_points[2]])
    #         tgt_polygon = geometry.Polygon([tgt_points[0], tgt_points[1], tgt_points[2]])
    #         # print(pre_points, tgt_points)
    #         # print(i)
    #         intersetion_area = pre_polygon.intersection(tgt_polygon).area
    #         union_area = pre_polygon.union(tgt_polygon).area
    #         iou_arr[i] = intersetion_area / union_area
    #         tgt_area_arr[i] = tgt_polygon.area
    #         pre_area_arr[i] = pre_polygon.area
    #         # co_loss_arr[i] = self._summary_object["ordered_best"][i]
    #         # wo_loss_arr[i] = self._summary_object["unordered_best"][i]
    #         # if True:
    #         #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 14))
    #         #
    #         #     ax1.fill(tgt_points.transpose()[0],tgt_points.transpose()[1], "b", pre_points.transpose()[0], pre_points.transpose()[1], "r", alpha=0.5)
    #         #     ax1.set_aspect(1.0)
    #         #     ax1.set_xlim(-20, 20)
    #         #     ax1.set_ylim(-20, 20)
    #         #
    #         #     ax2.set_title("F(phi)")
    #         #     ## target
    #         #     fc_arr_tgt = t2d.make_scatter_data(tgt_points, epsilon=0.002, dphi=0.001)
    #         #     ax2.plot(fc_arr_tgt[0], fc_arr_tgt[1], label="real_tgt")
    #         #     ax2.plot(fc_arr_tgt[0], fc_arr_tgt[2], label="imag_tgt")
    #         #     ## prediction
    #         #     fc_arr_pre = t2d.make_scatter_data(pre_points, epsilon=0.002, dphi=0.001)
    #         #     ax2.plot(fc_arr_pre[0], fc_arr_pre[1], label="real_pre")
    #         #     ax2.plot(fc_arr_pre[0], fc_arr_pre[2], label="imag_pre")
    #         #     ax2.legend(loc=4)
    #         #
    #         #
    #         #     ax1.set_title("(red) pre_points: p1={:2.2f},{:2.2f};p2={:2.2f},{:2.2f};p3={:2.2f},{:2.2f}\n"
    #         #                   "(blue)tgt_points: p1={:2.2f},{:2.2f};p2={:2.2f},{:2.2f};p3={:2.2f},{:2.2f}\n"
    #         #                   "iou: {:1.2f}; doa (real) {:1.2f}; doa (imag) {:1.2f}".format(
    #         #         pre_points[0][0], pre_points[0][1], pre_points[1][0], pre_points[1][1], pre_points[2][0], pre_points[2][1],
    #         #         tgt_points[0][0], tgt_points[0][1], tgt_points[1][0], tgt_points[1][1], tgt_points[2][0], tgt_points[2][1],
    #         #         intersetion_area / union_area, np.sum(np.abs(fc_arr_tgt[1] - fc_arr_pre[1])) / np.sum(np.abs(fc_arr_tgt[1]) + np.abs(fc_arr_pre[1])),
    #         #         np.sum(np.abs(fc_arr_tgt[2] - fc_arr_pre[2])) / np.sum(np.abs(fc_arr_tgt[2]) + np.abs(fc_arr_pre[2]))
    #         #     ))
    #         #     plt.grid()
    #         #     pdf = os.path.join(self._params['flags'].graph_dir, "single_plot_{}.pdf".format(sample_counter))
    #         #     sample_counter += 1
    #         #     fig.savefig(pdf)
    #         #     plt.clf()
    #         #     plt.close()
    #             # plt.show()
    #
    #     print("mean iou: {}".format(np.mean(iou_arr)))
    #     print("sum tgt area: {}; sum pre area: {}; p/t-area: {}".format(np.mean(tgt_area_arr), np.mean(pre_area_arr), np.sum(pre_area_arr) / np.sum(tgt_area_arr) ))
    #     # print("wrong order loss: {}; correct order loss: {}; order missed: {}".format(np.nanmean(wo_loss_arr),  np.nanmean(co_loss_arr), np.count_nonzero(~np.isnan(wo_loss_arr)) ))
    #
    #     from PyPDF2 import PdfFileMerger
    #
    #     plt.close("all")
    #     pdfs = [os.path.join(self._params['flags'].graph_dir, "single_plot_{}.pdf".format(x)) for x in range(sample_counter)]
    #     merger = PdfFileMerger()
    #     for pdf in pdfs:
    #         merger.append(pdf)
    #     merger.write(os.path.join(self._params['flags'].graph_dir, "plot_summary.pdf"))
    #     merger.close()
    #     for pdf in pdfs:
    #         if os.path.isfile(pdf):
    #             os.remove(pdf)
    #         else:
    #             logging.warning("Can not delete temporary file, result is probably incomplete!")
    #
