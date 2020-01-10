import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from shapely import geometry

import input_fn.input_fn_2d.data_gen_2dt.data_gen_t2d_util.polygone_2d_helper as old_helper

logger = logging.getLogger("polygone_2d_helper")
# logger.setLevel("DEBUG")
# logger.setLevel("INFO")

if __name__ == "__main__":
    logging.basicConfig()
    np.set_printoptions(precision=6, suppress=True)


class Fcalculator:
    def __init__(self, points, epsilon=np.array(0.0001), debug=False):
        """points is list of tupel with x,y like [(x1,y1), (x2,y2), (x3,y3),...]"""
        self.epsilon_tf = tf.constant(epsilon, dtype=tf.float64)
        self.points_tf = tf.Variable(points, dtype=tf.float64, trainable=False)
        self._debug = debug
        self._cross = tf.constant([[0.0, 1.0], [-1.0, 0.0]], dtype=tf.float64)

        self.epsilon = epsilon
        self.points = points

    def update_points(self, points):
        self.points_tf.assign(tf.cast(points, dtype=tf.float64))

    def q_of_phi(self, phi):
        phi_tf = tf.cast(phi, dtype=tf.float64)
        a__tf = tf.math.cos(phi_tf)
        b__tf = tf.math.sin(phi_tf) - 1.0
        q_tf = tf.stack([a__tf, b__tf])
        if self._debug:
            phi = np.array(phi, dtype=np.float64)
            a_ = np.cos(phi)
            b_ = np.sin(phi) - 1.0
            q = np.array([a_, b_])
            logger.debug("q^2: {}".format(tf.math.abs(q[0] ** 2 + q[1] ** 2)))
            assert np.array_equal(q, q_tf.numpy())
            return q
        else:
            return q_tf

    # def F_of_qs(self, q, p0_, p1_, c=0.0):
    #     p0 = np.array(p0_, dtype=np.float64)
    #     p0_tf = tf.Variable(p0_, dtype=tf.float64)
    #     p1 = np.array(p1_,  dtype=np.float64)
    #     p1_tf = tf.Variable(p1_, dtype=tf.float64)
    #     c = np.array(c)
    #     c_tf = tf.Variable(c, dtype=tf.float64)
    #     q_cross = np.array([-q[1], q[0]])
    #     q_cross_tf = tf.Variable([-q[1], q[0]])
    #     p0p1 = p1 - p0
    #     p0p1_tf = p1_tf - p0_tf
    #     scale = 1.0 / np.abs(np.abs(q[0] ** 2 + q[1] ** 2))
    #     scale_tf = 1.0 / tf.math.abs(tf.math.pow(tf.math.abs(q[0], 2) + tf.math.pow(q[1], 2)))
    #
    #     if scale >= 1000.0 / self.epsilon:
    #         logger.debug("Scale == NONE")
    #         polygon = geometry.Polygon(self.points)
    #         area = np.array(polygon.area, dtype=np.complex)
    #         logger.debug("area: {}".format(area))
    #         s_value = area / len(self.points)
    #     elif np.abs(np.dot(p0p1, q)) >= 0.0001:
    #         f_p0 = -1.0 * np.exp(1.0j * (np.dot(p0, q) + c))
    #         f_p1 = -1.0 * np.exp(1.0j * (np.dot(p1, q) + c))
    #         s_value = scale * np.dot(p0p1, q_cross) * (f_p1 - f_p0) / np.dot(p0p1, q)
    #     else:
    #         logger.debug("np.dot(p0p1, q) > epsilon")
    #         s_value = scale * np.dot(p0p1, q_cross) * -1.0j * np.exp(1.0j * (np.dot(p0, q) + c))
    #
    #     logger.debug("s_value: {:1.6f}".format(s_value))
    #     return s_value
    @tf.function
    def F_of_qs_arr(self, q, p0_, p1_, c=0.0):
        j_tf = tf.cast(tf.complex(0.0, 1.0), dtype=tf.complex128)
        p0_tf = tf.cast(p0_, dtype=tf.float64)
        p1_tf = tf.cast(p1_, dtype=tf.float64)
        q_tf = tf.cast(q, dtype=tf.float64)
        c_tf = tf.cast(c, dtype=tf.float64)
        c_tfc = tf.cast(c, dtype=tf.complex128)

        # q_cross_tf = tf.Variable([-q_tf[1], q_tf[0]])
        q_cross_tf = tf.matmul(tf.cast([[0.0, -1.0],[1.0, 0.0]], dtype=tf.float64), q_tf)

        p0p1_tf = p1_tf - p0_tf
        scale_tf = tf.cast(1.0 / tf.math.abs(q_tf[0] ** 2 + q_tf[1] ** 2), dtype=tf.float64)
        f_p0_tf = -tf.cast(1.0, dtype=tf.complex128) * tf.math.exp(j_tf * (tf.cast(complex_dot(p0_tf, q_tf), dtype=tf.complex128) + c_tfc))
        f_p1_tf = -tf.cast(1.0, dtype=tf.complex128) * tf.math.exp(j_tf * (tf.cast(complex_dot(p1_tf, q_tf), dtype=tf.complex128) + c_tfc))
        case1_array_tf = tf.cast(scale_tf * complex_dot(p0p1_tf, q_cross_tf), dtype=tf.complex128) * (f_p1_tf - f_p0_tf) / tf.cast(complex_dot(p0p1_tf, q_tf), dtype=tf.complex128)
        case2_array_tf = tf.cast(scale_tf * complex_dot(p0p1_tf, q_cross_tf), dtype=tf.complex128) * -j_tf * tf.math.exp(j_tf * tf.cast(complex_dot(p0_tf, q_tf) + c_tf, dtype=tf.complex128))
        res_array_tf = tf.where(tf.math.abs(complex_dot(p0p1_tf, q_tf)) >= 0.0001, case1_array_tf, case2_array_tf)

        if not self._debug:
            return res_array_tf

        else:
            j_ = np.array(1.0j, dtype=np.complex128)
            p0 = np.array(p0_, dtype=np.float64)
            p1 = np.array(p1_, dtype=np.float64)
            q = np.array(q, dtype=np.float64)
            q_cross = np.array([-q[1], q[0]])
            c = np.array(c)
            scale = 1.0 / np.abs(q[0] ** 2 + q[1] ** 2)
            p0p1 = p1 - p0
            assert np.array_equal(p0, p0_tf.numpy())
            assert np.array_equal(p1, p1_tf.numpy())
            assert np.array_equal(q, q_tf.numpy())
            # print(q_cross, q_cross_tf.numpy())
            assert np.array_equal(q_cross, q_cross_tf.numpy())
            assert np.array_equal(c, c_tf.numpy())
            assert np.array_equal(scale, scale_tf.numpy())

            f_p0 = -np.array(1.0, dtype=np.complex128) * np.exp(j_ * (np.dot(p0, q) + c))
            f_p1 = -np.array(1.0, dtype=np.complex128) * np.exp(j_ * (np.dot(p1, q) + c))

            assert np.array_equal(f_p0, f_p0_tf.numpy())
            assert np.array_equal(f_p1, f_p1_tf.numpy())
            case1_array = np.array(scale * np.dot(p0p1, q_cross), dtype=np.complex128) * (f_p1 - f_p0) / tf.cast(np.dot(p0p1, q), dtype=np.complex128)
            case2_array = np.array(scale * np.dot(p0p1, q_cross), dtype=np.complex128) * -1.0j * np.exp(1.0j * (tf.cast(np.dot(p0, q), dtype=np.complex128) + c))
            # print("complex dot in",p0p1_tf,q_cross_tf )
            # print("complex dot out", complex_dot(p0p1_tf, q_cross_tf))
            # print(case1_array, case1_array_tf.numpy())
            assert np.array_equal(case1_array, case1_array_tf.numpy())
            assert np.array_equal(case2_array, case2_array_tf.numpy())
            # print("case1_array.shape", case1_array.shape)
            res_array = np.where(np.abs(np.dot(p0p1, q)) >= 0.0001, case1_array, case2_array)

            # if np.max(scale) >= 1000.0 / self.epsilon:
            #     logger.debug("Scale == NONE")
            #     polygon = geometry.Polygon(self.points)
            #     area = np.array(polygon.area, dtype=np.complex)
            #     logger.debug("area: {}".format(area))
            #     s_value = area / len(self.points)
            #     case3_array = np.ones_like(q[0]) * s_value
            #     res_array = np.where(scale >= 1000.0 / self.epsilon, case3_array, res_array)
            #
            # if tf.math.reduce_max(scale_tf) >= 1000.0 / self.epsilon_tf:
            #     logger.debug("Scale_tf == NONE")
            #     polygon = geometry.Polygon(self.points_tf)
            #     area = np.array(polygon.area, dtype=np.complex)
            #     logger.debug("area: {}".format(area))
            #     s_value = area / len(self.points)
            #     case3_array = np.ones_like(q[0]) * s_value
            #     res_array_tf = np.where(scale >= 1000.0 / self.epsilon, case3_array, res_array)
            # print("res_array", res_array)
            # print("res_array_tf", res_array_tf)
            assert np.array_equal(res_array, res_array_tf.numpy())
            return res_array

    @tf.function
    def F_of_phi(self, phi):
        logger.debug("###########################################")
        logger.info("phi: {}".format(phi))
        q = self.q_of_phi(phi)
        c = 0.0
        if not self._debug:
            sum_res = tf.cast(phi * tf.cast(0.0, dtype=tf.float64), dtype=tf.complex128)
            c = tf.cast(c, dtype=tf.float64)
            for index in range(self.points_tf.shape[0]):
                p0 = self.points_tf[index - 1]
                p1 = self.points_tf[index]
                sum_res += self.F_of_qs_arr(q, p0, p1, c=c)

        else:
            sum_res = tf.zeros_like(phi, dtype=np.complex128)
            for index in range(len(self.points)):
                logger.debug("index: {}".format(index))
                p0 = self.points[index - 1]
                p1 = self.points[index]
                logger.debug("p0: {}; p1: {}".format(p0, p1))
                sum_res += self.F_of_qs_arr(q, p0, p1, c=c)
                logger.debug("sum_res {}".format(sum_res))

        final_res = sum_res
        logger.debug("sum_res.dtype: {}".format(sum_res.dtype))
        logger.info("final value: {}".format(final_res))
        return final_res


def complex_dot(a, b):
    return tf.einsum('i,i...->...', a, b)


if __name__ == "__main__":
    DEBUG = True
    DEBUG = False
    if DEBUG:
        logging.warning("DEBUG-MODE ON; GRAPH-MODE IS DISABLED!")
        tf.config.experimental_run_functions_eagerly(run_eagerly=True)
    t1 = time.time()

    for target in range(100):
        print(target)
        convex_polygon_arr = old_helper.generate_target_polygon(max_edge=3)
        convex_polygon_tuple = old_helper.array_to_tuples(convex_polygon_arr)
        polygon_calculator = Fcalculator(points=convex_polygon_tuple, debug=DEBUG)

        # phi_array = np.arange(np.pi / 2 - 1.5, np.pi / 2 + 1.5, 0.01)
        dphi = 0.0001
        har = 1.0 / 180.0 * np.pi  # hole_half_angle_rad
        mac = 1.0 / 180.0 * np.pi  # max_angle_of_view_cut_rad
        phi_array = np.concatenate((np.arange(0 + har, np.pi / 2 - mac, dphi),
                                  np.arange(np.pi / 2 + har, np.pi - mac, dphi)))
        if not DEBUG:
            phi_array = tf.cast(phi_array, dtype=tf.float64)
        # polygon_scatter_res = np.array(
        #     [polygon_calculator.F_of_phi(phi=phi).astype(dtype=np.complex64) for phi in phi_array])

        polygon_scatter_res = polygon_calculator.F_of_phi(phi=phi_array)
        if isinstance(polygon_scatter_res, tf.Tensor):
            polygon_scatter_res = polygon_scatter_res.numpy().astype(dtype=np.complex64)
        else:
            polygon_scatter_res = polygon_scatter_res.astype(dtype=np.complex64)

        # print(convex_polygon_arr.shape)
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 14))
        # ax1.plot(phi_array, polygon_scatter_res.real, "+b", label="real_polygon")
        # ax1.plot(phi_array, polygon_scatter_res.imag, "+r", label="imag_polygon")
        # ax1.plot(phi_array, np.abs(polygon_scatter_res), "+y", label="abs_polygon")
        # ax2.fill(convex_polygon_arr.transpose()[0], convex_polygon_arr.transpose()[1])
        # ax2.set_xlim((-50, 50))
        # ax2.set_ylim((-50, 50))
        # ax2.set_aspect(aspect=1.0)
        # plt.show()
    print("Time: {:0.1f}".format(time.time()-t1))

