import logging

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
    def __init__(self, points, epsilon=np.array(0.0001)):
        """points is list of tupel with x,y like [(x1,y1), (x2,y2), (x3,y3),...]"""
        self.epsilon = epsilon
        self.points = points

    def q_of_phi(self, phi):
        a_ = tf.math.cos(phi)
        b_ = tf.math.sin(phi) - 1.0
        q = tf.Variable([a_, b_])
        logger.debug("q^2: {}".format(tf.math.abs(q[0] ** 2 + q[1] ** 2)))
        return q

    def F_of_qs(self, q, p0_, p1_, c=0.0):
        p0 = np.array(p0_)
        p1 = np.array(p1_)
        c = np.array(c)
        q_cross = np.array([-q[1], q[0]])
        p0p1 = p1 - p0
        scale = 1.0 / np.abs(np.abs(q[0] ** 2 + q[1] ** 2))

        if scale >= 1000.0 / self.epsilon:
            logger.debug("Scale == NONE")
            polygon = geometry.Polygon(self.points)
            area = np.array(polygon.area, dtype=np.complex)
            logger.debug("area: {}".format(area))
            s_value = area / len(self.points)
        elif np.abs(np.dot(p0p1, q)) >= 0.0001:
            f_p0 = -1.0 * np.exp(1.0j * (np.dot(p0, q) + c))
            f_p1 = -1.0 * np.exp(1.0j * (np.dot(p1, q) + c))
            s_value = scale * np.dot(p0p1, q_cross) * (f_p1 - f_p0) / np.dot(p0p1, q)
        else:
            logger.debug("np.dot(p0p1, q) > epsilon")
            s_value = scale * np.dot(p0p1, q_cross) * -1.0j * np.exp(1.0j * (np.dot(p0, q) + c))

        logger.debug("s_value: {:1.6f}".format(s_value))
        return s_value

    def F_of_qs_arr(self, q, p0_, p1_, c=0.0):
        p0 = np.array(p0_)
        p1 = np.array(p1_)
        c = np.array(c)
        q_cross = np.array([-q[1], q[0]])
        p0p1 = p1 - p0
        # scale = 1.0 / np.abs(np.dot(q, q))
        scale = 1.0 / np.abs(q[0] ** 2 + q[1] ** 2)

        f_p0 = -tf.complex(1.0, 0.0) * tf.math.exp(tf.complex(0.0, 1.0) * tf.complex(tf.tensordot(p0, q, axes=0), 0.0))
        f_p1 = -tf.complex(1.0, 0.0) * tf.math.exp(tf.complex(0.0, 1.0) * tf.complex(tf.tensordot(p1, q, axes=0), 0.0))

        case1_array = scale * np.dot(p0p1, q_cross) * (f_p1 - f_p0) / np.dot(p0p1, q)
        case2_array = scale * np.dot(p0p1, q_cross) * -tf.complex(0.0, 1.0j) * tf.math.exp(
            tf.complex(0, 1.0j) * (np.dot(p0, q) + c))
        # print("case1_array.shape", case1_array.shape)
        res_array = np.where(np.abs(np.dot(p0p1, q)) >= 0.0001, case1_array, case2_array)

        if np.max(scale) >= 1000.0 / self.epsilon:
            logger.debug("Scale == NONE")
            polygon = geometry.Polygon(self.points)
            area = np.array(polygon.area, dtype=np.complex)
            logger.debug("area: {}".format(area))
            s_value = area / len(self.points)
            case3_array = np.ones_like(q[0]) * s_value
            res_array = np.where(scale >= 1000.0 / self.epsilon, case3_array, res_array)

        return res_array

    def F_of_phi(self, phi, c=0.0):
        logger.debug("###########################################")
        logger.info("phi: {}".format(phi))
        sum_res = np.zeros_like(phi, dtype=np.complex256)
        q = self.q_of_phi(phi)

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


if __name__ == "__main__":
    for target in range(4999):
        convex_polygon_arr = old_helper.generate_target_polygon()
        convex_polygon_tuple = old_helper.array_to_tuples(convex_polygon_arr)
        polygon_calculator = Fcalculator(points=convex_polygon_tuple)

        phi_array = np.arange(np.pi / 2 - 1.5, np.pi / 2 + 1.5, 0.01)
        polygon_scatter_res = np.array(
            [polygon_calculator.F_of_phi(phi=phi).astype(dtype=np.complex64) for phi in phi_array])

        print(convex_polygon_arr.shape)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 14))
        ax1.plot(phi_array, polygon_scatter_res.real, "-b", label="real_polygon")
        ax1.plot(phi_array, polygon_scatter_res.imag, "-y", label="imag_polygon")
        ax1.plot(phi_array, np.abs(polygon_scatter_res), "-y", label="abs_polygon")
        ax2.fill(convex_polygon_arr.transpose()[0], convex_polygon_arr.transpose()[1])
        ax2.set_xlim((-50, 50))
        ax2.set_ylim((-50, 50))
        ax2.set_aspect(aspect=1.0)
        plt.show()
