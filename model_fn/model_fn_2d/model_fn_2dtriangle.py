import logging
import os
import shutil

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import input_fn.input_fn_2d.data_gen_2dt.data_gen_t2d_util.triangle_2d_helper as t2d
import model_fn.model_fn_2d.util_2d.graphs_2d as graphs
import model_fn.util_model_fn.custom_layers as c_layer
from model_fn.model_fn_base import ModelBase


class ModelTriangle(ModelBase):
    def __init__(self, params):
        super(ModelTriangle, self).__init__(params)
        self.mydtype = tf.float32
        self._targets = None
        self._point_dist = None
        self._summary_object = {"tgt_points": [], "pre_points": [], "ordered_best": [], "unordered_best": []}
        self._graph = self.get_graph()
        self._scatter_calculator = None
        self.scatter_polygon_tf = None
        self._loss = tf.Variable(0.0, dtype=self.mydtype, trainable=False)
        # log different log types to tensorboard:
        self.metrics["train"]["loss_input_diff"] = tf.keras.metrics.Mean("loss_input_diff", self.mydtype)
        self.metrics["eval"]["loss_point_diff"] = tf.keras.metrics.Mean("loss_point_diff", self.mydtype)
        self.metrics["train"]["loss_point_diff"] = tf.keras.metrics.Mean("loss_point_diff", self.mydtype)
        self.metrics["eval"]["loss_input_diff"] = tf.keras.metrics.Mean("loss_input_diff", self.mydtype)

    def set_interface(self, val_dataset):
        build_inputs, build_out = super(ModelTriangle, self).set_interface(val_dataset)
        # if self._flags.loss_mode == "input_diff":
        self.scatter_polygon_tf = c_layer.ScatterPolygonTF(fc_tensor=tf.cast(build_inputs[0]["fc"],
                                                                             dtype=self.mydtype),
                                                           points_tf=tf.cast(build_inputs[1]["points"],
                                                                             dtype=self.mydtype),
                                                           with_batch_dim=True, dtype=self.mydtype)
        return build_inputs, build_out

    def get_graph(self):
        return getattr(graphs, self._params['flags'].graph)(self._params)

    def get_placeholder(self):
        if not self._flags.complex_phi:
            return {"fc": tf.compat.v1.placeholder(tf.float32, [None, 3, None], name="infc")}
        else:
            return {"fc": tf.compat.v1.placeholder(tf.float32, [None, 4, None], name="infc")}

    def get_output_nodes(self, has_graph=True):
        if has_graph:
            tf.identity(self._graph_out['pre_points'], name="pre_points")  # name to grab from java
        return "pre_points"  # return names as comma separated string without spaces

    def get_target_keys(self):
        return 'points'

    def get_predictions(self):
        return self._graph_out['pre_points']

    def info(self):
        self.get_graph().print_params()

    def loss(self, predictions, targets):
        self._loss = tf.constant(0.0, dtype=self.mydtype)
        fc = tf.cast(predictions['fc'], dtype=self.mydtype)
        pre_points = tf.cast(tf.reshape(predictions['pre_points'], [-1, 3, 2]), dtype=self.mydtype)
        res_scatter = self.scatter_polygon_tf(points_tf=pre_points)
        loss_input_diff = tf.reduce_mean(tf.keras.losses.mean_squared_error(res_scatter, fc[:, 1:, :]))
        loss_point_diff = tf.reduce_mean(tf.keras.losses.mean_squared_error(pre_points, targets["points"]))
        # tf.print("input_diff-loss", loss_input_diff)
        if self._flags.loss_mode == "input_diff":
            self._loss += loss_input_diff
        elif self._flags.loss_mode == "point_diff":
            self._loss += loss_point_diff

        self.metrics[self._mode]["loss_input_diff"](loss_input_diff)
        self.metrics[self._mode]["loss_point_diff"](loss_point_diff)

        # plt.figure()
        # plt.plot(fc[0, 0, :], fc[0, 1, :], label="input_real")
        # plt.plot(fc[0, 0, :], fc[0, 2, :], label="input_imag")
        # plt.plot(fc[0, 0, :], res_scatter[0, 0, :], label="pred_rec_real")
        # plt.plot(fc[0, 0, :], res_scatter[0, 1, :], label="pred_rec_imag")
        # res_scatter = self.scatter_polygon_tf(points_tf=targets["points"])
        # plt.plot(fc[0, 0, :], res_scatter[0, 0, :], label="input_rec_real")
        # plt.plot(fc[0, 0, :], res_scatter[0, 1, :], label="input_rec_imag")
        # print("phi:", fc[0, 0, :])
        # print("tgt shape", tf.shape(targets["points"]))
        # print("scatter res shape", tf.shape(res_scatter))
        # plt.legend()
        # plt.show()
        # if self._flags.loss_mode == "point3":
        #     loss0 = tf.cast(batch_point3_loss(self._targets['points'], self._graph_out['pre_points'],
        #                                   self._params["flags"].train_batch_size), dtype=tf.float32)
        # elif self._flags.loss_mode == "no_match":
        #     loss0 = tf.cast(ordered_point3_loss(self._targets['points'], self._graph_out['pre_points'],
        #                                   self._params["flags"].train_batch_size, keras_graph=self), dtype=tf.float32)
        # elif self._flags.loss_mode == "fast_no_match":
        #     loss0 = tf.cast(no_match_loss(self._targets['points'], self._graph_out['pre_points'],
        #                                   self._params["flags"].train_batch_size), dtype=tf.float32)
        # else:
        #     raise KeyError("Loss-mode: {} do not exist!".format(self._flags.loss_mode))

        return self._loss

    def export_helper(self):
        for train_list in self._params['flags'].train_lists:
            data_id = os.path.basename(train_list).replace("_train.lst", "").replace("_val.lst", "")
            shutil.copy(os.path.join("data/synthetic_data", data_id, "log_{}_train.txt".format(data_id)),
                        os.path.join(self._params['flags'].checkpoint_dir, "export"))
        data_id = os.path.basename(self._params['flags'].val_list).replace("_train.lst", "").replace("_val.lst", "")
        shutil.copy(os.path.join("data/synthetic_data", data_id, "log_{}_val.txt".format(data_id)),
                    os.path.join(self._params['flags'].checkpoint_dir, "export"))

    def print_evaluate(self, output_dict, target_dict):
        with tf.compat.v1.Session().as_default():
            tgt_area_sum = 0
            area_diff_sum = 0
            # loss_ordered = ordered_point3_loss(output_dict["pre_points"], target_dict["points"], self._params['flags'].val_batch_size)
            # loss_best = batch_point3_loss(output_dict["pre_points"], target_dict["points"], self._params['flags'].val_batch_size)

            self._summary_object["tgt_points"].extend(
                [target_dict["points"][x] for x in range(self._params['flags'].val_batch_size)])
            self._summary_object["pre_points"].extend(
                [output_dict["pre_points"][x] for x in range(self._params['flags'].val_batch_size)])
            ob_buffer_list = []
            ub_buffer_list = []
            # for i in range(output_dict["pre_points"].shape[0]):
            # print("## {:4d} Sample ##".format(i))
            # # print(loss_ordered.eval())
            # print("loss: {:3.2f}(ordered)| {:3.2f} (best)".format(loss_ordered.eval()[i], loss_best.eval()[i]))
            # if np.abs(loss_ordered.eval()[i] - loss_best.eval()[i]) > 0.01:
            #     # print("WARNING: losses are not equal")
            #     ob_buffer_list.append(np.nan)
            #     ub_buffer_list.append(loss_best.eval()[i])
            # else:
            #     ob_buffer_list.append(loss_best.eval()[i])
            #     ub_buffer_list.append(np.nan)

            self._summary_object["ordered_best"].extend(ob_buffer_list)
            self._summary_object["unordered_best"].extend(ub_buffer_list)

            # print("predicted points")
            # print(output_dict["pre_points"][i])
            # print("target points")
            # print(target_dict["points"][i])
            # pred_area = np.abs(np.dot((output_dict["pre_points"][i][0] - output_dict["pre_points"][i][1]), (output_dict["pre_points"][i][1] - output_dict["pre_points"][i][2])) / 2.0)
            # tgt_area = np.abs(np.dot((target_dict["points"][i][0] - target_dict["points"][i][1]), (target_dict["points"][i][1] - target_dict["points"][i][2])) / 2.0)
            # area_diff_sum += np.max(pred_area - tgt_area)
            # tgt_area_sum += tgt_area
            # print("area diff: {:0.3f}".format(np.abs(pred_area - tgt_area) / tgt_area))
            # print("target area: {:0.3f}".format(np.abs(tgt_area)))

        return area_diff_sum, tgt_area_sum

    def print_evaluate_summary(self):
        sample_counter = 0
        import matplotlib.pyplot as plt

        from shapely import geometry
        summary_lenght = len(self._summary_object["tgt_points"])
        print("summary length: {}".format(summary_lenght))

        tgt_area_arr = np.zeros(summary_lenght)
        pre_area_arr = np.zeros(summary_lenght)
        pre_area_arr = np.zeros(summary_lenght)
        iou_arr = np.zeros(summary_lenght)
        co_loss_arr = np.ones(summary_lenght) * np.nan
        wo_loss_arr = np.ones(summary_lenght) * np.nan

        for i in range(summary_lenght):
            pre_points = np.reshape(self._summary_object["pre_points"][i], (3, 2))
            tgt_points = np.reshape(self._summary_object["tgt_points"][i], (3, 2))
            # print(pre_points)
            # print(tgt_points)
            pre_polygon = geometry.Polygon([pre_points[0], pre_points[1], pre_points[2]])
            tgt_polygon = geometry.Polygon([tgt_points[0], tgt_points[1], tgt_points[2]])
            # print(pre_points, tgt_points)
            # print(i)
            intersetion_area = pre_polygon.intersection(tgt_polygon).area
            union_area = pre_polygon.union(tgt_polygon).area
            iou_arr[i] = intersetion_area / union_area
            tgt_area_arr[i] = tgt_polygon.area
            pre_area_arr[i] = pre_polygon.area
            # co_loss_arr[i] = self._summary_object["ordered_best"][i]
            # wo_loss_arr[i] = self._summary_object["unordered_best"][i]
            PLOT = True
            if PLOT:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 14))
                # fig = plt.figure(figsize=(38, 14))
                # ax1 = fig.add_subplot(121)
                # ax2 = fig.add_subplot(122, projection='3d')

                ax1.fill(tgt_points.transpose()[0], tgt_points.transpose()[1], "b", pre_points.transpose()[0],
                         pre_points.transpose()[1], "r", alpha=0.5)
                ax1.set_aspect(1.0)
                ax1.set_xlim(-50, 50)
                ax1.set_ylim(-50, 50)

                ax2.set_title("F(phi)")
                dphi = 0.01
                har = 00.0 / 180.0 * np.pi  # hole_half_angle_rad
                mac = 00.0 / 180.0 * np.pi  # max_angle_of_view_cut_rad
                phi_arr = np.concatenate((np.arange(0 + har, np.pi / 2 - mac, dphi),
                                          np.arange(np.pi / 2 + har, np.pi - mac, dphi)))
                # # target
                # fc_arr_tgt = t2d.make_scatter_data(tgt_points, epsilon=0.002, phi_arr=phi_arr)
                # ax2.plot(fc_arr_tgt[0], fc_arr_tgt[1], label="real_tgt")
                # ax2.plot(fc_arr_tgt[0], fc_arr_tgt[2], label="imag_tgt")
                # ## prediction
                # fc_arr_pre = t2d.make_scatter_data(pre_points, epsilon=0.002, phi_arr=phi_arr)
                # ax2.plot(fc_arr_pre[0], fc_arr_pre[1], label="real_pre")
                # ax2.plot(fc_arr_pre[0], fc_arr_pre[2], label="imag_pre")
                # ax2.legend(loc=4)

                ## target scatter
                # phi_3dim = np.abs(phi_arr - np.pi / 2)
                # fc_arr_tgt = t2d.make_scatter_data(tgt_points, phi_arr=phi_arr, epsilon=0.002)
                # ## prediction
                # fc_arr_pre = t2d.make_scatter_data(pre_points, phi_arr=phi_arr, epsilon=0.002)
                # for idx in range(fc_arr_tgt[2].shape[0]):
                #     ax2.plot((fc_arr_tgt[1][idx], fc_arr_pre[1][idx]), (fc_arr_tgt[2][idx], fc_arr_pre[2][idx]),phi_3dim[idx],  label="diffs")
                # #     # ax2.plot(fc_arr_tgt[1], fc_arr_tgt[2], phi_3dim,  'b', label="diffs", linewidth=0.2)
                # #     # ax2.plot(fc_arr_pre[1], fc_arr_pre[2], phi_3dim,  'r', label="diffs", linewidth=0.2)
                # #     ax2.plot(fc_arr_tgt[1], fc_arr_tgt[2], 'b', label="diffs", linewidth=0.2)
                # #     ax2.plot(fc_arr_pre[1], fc_arr_pre[2], 'r', label="diffs", linewidth=0.2)
                #     ax2.set_xlabel("real")
                #     ax2.set_ylabel("imag")
                # #     # ax2.set_zlabel("|phi-pi/2|")

                ## complexphi
                # target
                range_arr = (np.arange(10, dtype=self.mydtype) + 1.0) / 10.0
                zeros_arr = np.zeros_like(range_arr, dtype=self.mydtype)
                a = np.concatenate((range_arr, range_arr, zeros_arr), axis=0)
                b = np.concatenate((zeros_arr, range_arr, range_arr), axis=0)
                phi_arr = a + 1.0j * b
                fc_arr_tgt = t2d.make_scatter_data(tgt_points, phi_arr=phi_arr, epsilon=0.002, dphi=0.001,
                                                   complex_phi=True)
                fc_arr_pre = t2d.make_scatter_data(pre_points, phi_arr=phi_arr, epsilon=0.002, dphi=0.001,
                                                   complex_phi=True)
                # ax2.scatter(fc_arr_tgt[0], fc_arr_tgt[1], label="real_tgt")
                for idx in range(fc_arr_tgt[2].shape[0]):
                    ax2.plot((fc_arr_tgt[2][idx], fc_arr_pre[2][idx]), (fc_arr_pre[3][idx], fc_arr_tgt[3][idx]),
                             label="diffs")

                # ax2.legend(loc=4)

                ## for beamer neiss-kick of 2019/10/30
                # from mpl_toolkits.axes_grid1 import Divider, Size
                # h = [Size.Fixed(0.0), Size.Scaled(0.), Size.Fixed(.0)]
                # v = [Size.Fixed(0.0), Size.Scaled(0.), Size.Fixed(.0)]
                # fig = plt.figure(figsize=(10, 4))
                # divider = Divider(fig, (0.06, 0.13, 0.35, 0.80), h, v, aspect=False)
                # divider2 = Divider(fig, (0.55, 0.13, 0.4, 0.80), h, v, aspect=False)
                # ax1 = plt.Axes(fig, divider.get_position())
                # ax2 = plt.Axes(fig, divider2.get_position())
                # ax1.annotate('', xy=(1.32, 0.3), xycoords='axes fraction', xytext=(1.05, 0.3),
                #                 arrowprops=dict(headlength=12, headwidth=12, color='b', lw=8))
                # ax1.annotate('', xy=(1.05, 0.6), xycoords='axes fraction', xytext=(1.326, 0.6),
                #                 arrowprops=dict(headlength=12, headwidth=12, color='r', lw=8))
                # ax1.annotate('', xy=(1.32, 0.1), xycoords='axes fraction', xytext=(1.05, 0.1),
                #                 arrowprops=dict(headlength=12, headwidth=12, color='g', lw=8))
                # fig.text(0.4, 0.42, "Berechnung",)
                # fig.text(0.4, 0.67, "Neuronales Netz",)
                # fig.text(0.4, 0.25, "Vergleich",)
                # fig.add_axes(ax1)
                # fig.add_axes(ax2)
                # plt.rc('text', usetex=True,)
                # plt.rc('font', family='serif', size=14)
                # ax1.fill(tgt_points.transpose()[0], tgt_points.transpose()[1], "b", label="Ziel", alpha=0.5)
                # ax1.fill(pre_points.transpose()[0], pre_points.transpose()[1], "r", label="Netz", alpha=0.5)
                # fig.text(0.34, 0.78, "Ziel",bbox=dict(facecolor='blue', alpha=0.5))
                # fig.text(0.34, 0.86, "Netz",bbox=dict(facecolor='red', alpha=0.5))
                # ax1.set_aspect(1.0)
                # ax1.set_xticks([-10, 0, 10])
                # ax1.set_yticks([-10, 0, 10])
                # ax1.set_xlim(-17, 17)
                # ax1.set_ylim(-17, 17)
                # ax1.set_ylabel(r'y', usetex=True)
                # ax1.set_xlabel(r'x', usetex=True)
                # ax2.set_ylabel(r'F', usetex=True)
                # ax2.set_xlabel(r'$\varphi$', usetex=True)
                # ## target
                # fc_arr_tgt = t2d.make_scatter_data(tgt_points, epsilon=0.002, dphi=0.001)
                # fc_arr_pre = t2d.make_scatter_data(pre_points, epsilon=0.002, dphi=0.001)
                # norm = np.max(np.concatenate([fc_arr_tgt[1], fc_arr_tgt[2], fc_arr_pre[1], fc_arr_pre[2] ]))
                # ax2.plot(fc_arr_tgt[0], fc_arr_tgt[1]/norm, label="Re(F$_\mathrm{Ziel}$)")
                # ax2.plot(fc_arr_tgt[0], fc_arr_tgt[2]/norm, label="Im(F$_\mathrm{Ziel}$)")
                # ## prediction
                # ax2.plot(fc_arr_pre[0], fc_arr_pre[1]/norm, label="Re(F$_\mathrm{Netz}$)")
                # ax2.plot(fc_arr_pre[0], fc_arr_pre[2]/norm, label="Im(F$_\mathrm{Netz}$)")
                # ax2.set_yticks([0, 1])
                # ax2.set_xticks([0, np.pi/2, np.pi])
                # ax2.set_xticklabels(["0", "$\pi/2$", "$\pi$"])
                # ax2.legend(loc=2)

                ax1.set_title("(red) pre_points: p1={:2.2f},{:2.2f};p2={:2.2f},{:2.2f};p3={:2.2f},{:2.2f}\n"
                              "(blue)tgt_points: p1={:2.2f},{:2.2f};p2={:2.2f},{:2.2f};p3={:2.2f},{:2.2f}\n"
                              "iou: {:1.2f}; doa (real) {:1.2f}; doa (imag) {:1.2f}".format(
                    pre_points[0][0], pre_points[0][1], pre_points[1][0], pre_points[1][1], pre_points[2][0],
                    pre_points[2][1],
                    tgt_points[0][0], tgt_points[0][1], tgt_points[1][0], tgt_points[1][1], tgt_points[2][0],
                    tgt_points[2][1],
                    intersetion_area / union_area, np.sum(np.abs(fc_arr_tgt[1] - fc_arr_pre[1])) / np.sum(
                        np.abs(fc_arr_tgt[1]) + np.abs(fc_arr_pre[1])),
                    np.sum(np.abs(fc_arr_tgt[2] - fc_arr_pre[2])) / np.sum(
                        np.abs(fc_arr_tgt[2]) + np.abs(fc_arr_pre[2]))
                ))
                plt.grid()
                pdf = os.path.join(self._params['flags'].model_dir, "single_plot_{}.pdf".format(sample_counter))
                # svg = os.path.join(self._params['flags'].model_dir, "single_plot_{}.svg".format(sample_counter))
                sample_counter += 1
                fig.savefig(pdf)
                # plt.show()
                plt.clf()
                plt.close()

        print("mean iou: {}".format(np.mean(iou_arr)))
        print("sum tgt area: {}; sum pre area: {}; p/t-area: {}".format(np.mean(tgt_area_arr), np.mean(pre_area_arr),
                                                                        np.sum(pre_area_arr) / np.sum(tgt_area_arr)))
        # print("wrong order loss: {}; correct order loss: {}; order missed: {}".format(np.nanmean(wo_loss_arr),  np.nanmean(co_loss_arr), np.count_nonzero(~np.isnan(wo_loss_arr)) ))

        if PLOT:
            from PyPDF2 import PdfFileMerger

            plt.close("all")
            pdfs = [os.path.join(self._params['flags'].model_dir, "single_plot_{}.pdf".format(x)) for x in
                    range(sample_counter)]
            merger = PdfFileMerger()
            for pdf in pdfs:
                merger.append(pdf)
            merger.write(os.path.join(self._params['flags'].model_dir, "plot_summary.pdf"))
            merger.close()
            for pdf in pdfs:
                if os.path.isfile(pdf):
                    os.remove(pdf)
                else:
                    logging.warning("Can not delete temporary file, result is probably incomplete!")
