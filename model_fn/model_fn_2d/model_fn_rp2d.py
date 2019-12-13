import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from shapely import geometry
from itertools import permutations

import input_fn.input_fn_2d.data_gen_2dt.data_gen_t2d_util.polygone_2d_helper as p2dh
import model_fn.model_fn_2d.util_2d.graphs_rp2d as graphs
from model_fn.model_fn_base import ModelBase


class ModelRegularPolygon(ModelBase):
    def __init__(self, params):
        super(ModelRegularPolygon, self).__init__(params)
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
            tf.identity(self._graph_out['radius_pred'], name="radius_pred")  # name to grab from java
            tf.identity(self._graph_out['rotation_pred'], name="rotation_pred")  # name to grab from java
            tf.identity(self._graph_out['translation_pred'], name="translation_pred")  # name to grab from java
            tf.identity(self._graph_out['edges_pred'], name="edges_pred")  # name to grab from java
        return "radius_pred,rotation_pred,translation_pred,edges_pred"  # return names as comma separated string without spaces

        #
        # return {"radius_pred": radius_final,
        #         "rotation_pred": rotation_final,
        #         "translation_pred": translation_final,
        #         # "edges_pred": edge_final}

    def get_target_keys(self):
        return 'radius,rotation,translation,edges'

    def get_predictions(self):
        return self._graph_out

    def info(self):
        self.get_graph().print_params()

    def get_loss(self):
        # self._targets['points'] = tf.Print(self._targets['points'], [self._targets['points']])
        # loss0 = tf.losses.absolute_difference(self._targets['points'], self._graph_out['p_pred'])
        # print("params train batch size", self._params["flags"].train_batch_size)
        # print("points", self._targets['points'])
        max_edges = self._flags.max_edges
        loss = 0.0

        loss_edge = tf.reduce_mean(tf.sqrt(tf.compat.v1.losses.softmax_cross_entropy(
            tf.one_hot(tf.squeeze(self._targets['edges'], axis=-1) - 3, depth=max_edges - 3),
            self._graph_out['edges_pred'])), name="loss_edge")
        loss_radius = tf.losses.mean_squared_error(self._targets['radius'], self._graph_out["radius_pred"],
                                                   scope="loss_radius")
        loss_rotation = tf.losses.mean_squared_error(self._targets['rotation'], self._graph_out["rotation_pred"],
                                                     scope="loss_rotation")
        loss_translation = tf.losses.mean_squared_error(
            tf.sqrt(tf.reduce_sum(tf.math.square(self._targets['translation']))),
            tf.sqrt(tf.reduce_sum(tf.math.square(self._graph_out["translation_pred"]))), scope="loss_translation")

        loss = loss_edge + loss_radius + loss_rotation + loss_translation / 10.0
        # loss = tf.Print(loss, [loss, loss_edge, loss_radius, loss_rotation, loss_translation], message="loss:all, edge, radius, rotation, translation:")

        return loss

    def export_helper(self):
        for train_list in self._params['flags'].train_lists:
            data_id = os.path.basename(train_list)[:-10]
            shutil.copy(os.path.join("data/synthetic_data", data_id, "log_{}_train.txt".format(data_id)),
                        os.path.join(self._params['flags'].checkpoint_dir, "export"))
        data_id = os.path.basename(self._params['flags'].val_list)[:-8]
        shutil.copy(os.path.join("data/synthetic_data", data_id, "log_{}_val.txt".format(data_id)),
                    os.path.join(self._params['flags'].checkpoint_dir, "export"))

    #
    # def get_metrics(self):
    #     loss_rotation = tf.get_variable(name="graph/loss_rotation_0")
    #     return {'loss_rotaion': loss_rotation}

    def print_evaluate(self, output_dict, target_dict):
        with tf.compat.v1.Session().as_default():
            tgt_area_sum = 0
            area_diff_sum = 0
            print("Targets")
            print(target_dict)
            print("Predictions:")
            print(output_dict)
            iou_arr = np.zeros(output_dict["edges_pred"].shape[0])
            for i in range(output_dict["edges_pred"].shape[0]):
                print(output_dict['edges_pred'][i])
                rpf_dict_pred = {'radius': float(output_dict['radius_pred'][i]),
                                 'rotation': float(output_dict['rotation_pred'][i]),
                                 'translation': np.squeeze(output_dict['translation_pred'][i]),
                                 'edges': np.argmax(output_dict['edges_pred'][i]) + 3}
                rpf_dict_tgt = {'radius': float(target_dict['radius'][i]),
                                'rotation': float(target_dict['rotation'][i]),
                                'translation': np.squeeze(target_dict['translation'][i]),
                                'edges': int(target_dict['edges'][i])}
                pred_array = p2dh.rpf_to_points_array(rpf_dict_pred)
                pred_tuples = p2dh.array_to_tuples(pred_array)
                tgt_array = p2dh.rpf_to_points_array(rpf_dict_tgt)
                tgt_tuples = p2dh.array_to_tuples(tgt_array)
                pre_polygon = geometry.Polygon(pred_tuples)
                tgt_polygon = geometry.Polygon(tgt_tuples)
                intersetion_area = pre_polygon.intersection(tgt_polygon).area
                union_area = pre_polygon.union(tgt_polygon).area
                iou_arr[i] = intersetion_area / union_area

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 14))
                ## prediction
                polygon_calculator = p2dh.Fcalculator(pred_tuples)
                phi_array = np.arange(np.pi / 2 - 1.5, np.pi / 2 + 1.5, 0.01)
                polygon_scatter_res = polygon_calculator.F_of_phi(phi=phi_array).astype(dtype=np.complex64)
                ax1.fill(pred_array.transpose()[0], pred_array.transpose()[1], label="pred", alpha=0.5)
                ax2.plot(phi_array, polygon_scatter_res.real, "-b", label="real_polygon_pred")
                ax2.plot(phi_array, polygon_scatter_res.imag, "-y", label="imag_polygon_pred")
                ax2.plot(phi_array, np.abs(polygon_scatter_res), "-r", label="abs_polygon_pred")

                ## target
                polygon_calculator = p2dh.Fcalculator(tgt_tuples)
                phi_array = np.arange(np.pi / 2 - 1.5, np.pi / 2 + 1.5, 0.01)
                polygon_scatter_res = polygon_calculator.F_of_phi(phi=phi_array).astype(dtype=np.complex64)
                ax1.fill(tgt_array.transpose()[0], tgt_array.transpose()[1], label="tgt", alpha=0.5)
                ax2.plot(phi_array, polygon_scatter_res.real, "-b", label="real_polygon_tgt")
                ax2.plot(phi_array, polygon_scatter_res.imag, "-y", label="imag_polygon_tgt")
                ax2.plot(phi_array, np.abs(polygon_scatter_res), "-r", label="abs_polygon_tgt")

                ax1.set_xlim((-50, 50))
                ax1.set_ylim((-50, 50))
                ax1.set_aspect(aspect=1.0)
                ax2.legend(loc=2)
                ax1.legend(loc=2)
                plt.show()
                plt.clf()
                plt.close()

                print("iou:", iou_arr[i])

        print("mean iou:", np.mean(iou_arr))

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
    #         #     pdf = os.path.join(self._params['flags'].model_dir, "single_plot_{}.pdf".format(sample_counter))
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
    #     pdfs = [os.path.join(self._params['flags'].model_dir, "single_plot_{}.pdf".format(x)) for x in range(sample_counter)]
    #     merger = PdfFileMerger()
    #     for pdf in pdfs:
    #         merger.append(pdf)
    #     merger.write(os.path.join(self._params['flags'].model_dir, "plot_summary.pdf"))
    #     merger.close()
    #     for pdf in pdfs:
    #         if os.path.isfile(pdf):
    #             os.remove(pdf)
    #         else:
    #             logging.warning("Can not delete temporary file, result is probably incomplete!")
    #
