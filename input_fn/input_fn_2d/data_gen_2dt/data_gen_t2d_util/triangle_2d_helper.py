import logging

import numpy as np


class Fcalculator:
    def __init__(self, p1=np.array([0.0, 0.0]), p2=np.array([1.0, 0.0]), p3=np.array([0.0, 1.0]),
                 epsilon=np.array(0.0001), no_check=False, complex_phi=False):
        self._p1 = np.array(p1, dtype=np.float128)
        self._p2 = np.array(p2, dtype=np.float128)
        self._p3 = np.array(p3, dtype=np.float128)
        if not no_check:  # skip check if valid input is ensured for better performance
            assert np.sum(np.square(np.abs(self._p1 - self._p2))) > (10 * epsilon) ** 2
            assert np.sum(np.square(np.abs(self._p2 - self._p3))) > (10 * epsilon) ** 2
            assert np.sum(np.square(np.abs(self._p3 - self._p1))) > (10 * epsilon) ** 2

        self._jacobi_det = np.abs((self._p2[0] - self._p1[0]) * (self._p3[1] - self._p1[1]) -
                                  (self._p3[0] - self._p1[0]) * (self._p2[1] - self._p1[1]))
        self._epsilon = np.array(epsilon, dtype=np.float128)
        self._complex_phi = complex_phi

    @staticmethod
    def _case1(a, b):
        logging.info("case1, a!=b, a!=0, b!=0")
        with np.errstate(divide='ignore', invalid='ignore'):  # nan values not used by np.where-condition
            return 1.0 / (a * b * (b - a)) * (b * (np.exp(1.0j * a) - 1.0) - a * (np.exp(1.0j * b) - 1.0))

    @staticmethod
    def _case2(b):
        logging.info("case2, a!=b, a=0, b!=0")
        with np.errstate(divide='ignore', invalid='ignore'):  # nan values not used by np.where-condition
            return 1.0j / b - 1 / b ** 2 * (np.exp(1.0j * b) - 1.0)

    @staticmethod
    def _case3(a):
        logging.info("case3, a!=b, b=0, a!=0")
        with np.errstate(divide='ignore', invalid='ignore'):  # nan values not used by np.where-condition
            return 1.0j / a - 1 / a ** 2 * (np.exp(1.0j * a) - 1.0)

    @staticmethod
    def _case5(a):
        logging.info("case5, a=b, b!=0, a!=0")
        with np.errstate(divide='ignore', invalid='ignore'):  # nan values not used by np.where-condition
            return np.exp(1.0j * a) / (1.0j * a) + (np.exp(1.0j * a) - 1.0) / a ** 2

    def set_triangle(self, p1=np.array([0.0, 0.0], dtype=np.float128), p2=np.array([1.0, 0.0], dtype=np.float128),
                     p3=np.array([0.0, 1.0], dtype=np.float128), no_check=False):
        self.__init__(p1, p2, p3, self._epsilon, no_check)

    def call_on_array(self, phi_array):
        if not self._complex_phi:
            phi_array = np.array(phi_array, dtype=np.float128)
            a_ = np.cos(phi_array, dtype=np.float128)
            b_ = np.sin(phi_array, dtype=np.float128) - 1.0
        else:
            phi_array = np.array(phi_array, dtype=np.complex128)
            a_ = phi_array.real
            b_ = phi_array.imag
            # print("a_", a_)
            # print("b_", b_)

        a = a_ * (self._p2[0] - self._p1[0]) + b_ * (self._p2[1] - self._p1[1])
        b = a_ * (self._p3[0] - self._p1[0]) + b_ * (self._p3[1] - self._p1[1])
        c = a_ * self._p1[0] + b_ * self._p1[1]

        f_array = np.full_like(phi_array, np.nan, dtype=np.complex256)

        a_not_b = np.abs(a - b) > self._epsilon
        a_is_b = np.abs(a - b) <= self._epsilon
        a_not_0 = np.abs(a) - self._epsilon > 0
        b_not_0 = np.abs(b) - self._epsilon > 0
        a_is_0 = np.abs(a) <= self._epsilon
        b_is_0 = np.abs(b) <= self._epsilon

        cond1 = np.logical_and(np.logical_and(a_not_b, a_not_0), b_not_0)
        cond2 = np.logical_and(np.logical_and(a_not_b, a_is_0), b_not_0)
        cond3 = np.logical_and(np.logical_and(a_not_b, b_is_0), a_not_0)
        cond4 = np.logical_or(np.logical_or(np.logical_and(a_is_0, a_is_b), np.logical_and(b_is_0, a_is_b)),
                              np.logical_and(b_is_0, a_is_0))
        cond5 = np.logical_and(np.logical_and(a_is_b, b_not_0), a_not_0)
        assert (np.logical_xor(cond1, np.logical_xor(cond2, np.logical_xor(cond3, np.logical_xor(cond4,
                                                                                                 cond5))))).all() == True

        f_array = np.where(cond1, self._case1(a, b), f_array)
        f_array = np.where(cond2, self._case2(b), f_array)
        f_array = np.where(cond3, self._case3(a), f_array)
        f_array = np.where(cond4, 0.5, f_array)
        f_array = np.where(cond5, self._case5(a), f_array)

        assert np.isnan(f_array).any() == False
        return self._jacobi_det * np.exp(1.0j * c) * f_array


def multi_triangle_f(phi, p1=np.array([0.0, 0.0], dtype=np.float64), p2=np.array([1.0, 0.0], dtype=np.float64),
                     p3=np.array([0.0, 1.0], dtype=np.float64), epsilon=0.001,
                     no_check=False):
    """slow straight forward version of Fcalculator, for scalar values of phi only"""
    phi = np.array(phi, dtype=np.float64)
    if not no_check:  # skip check if valid input is ensured for better performance
        assert np.sum(np.square(np.abs(p1 - p2))) > (10 * epsilon) ** 2
        assert np.sum(np.square(np.abs(p2 - p3))) > (10 * epsilon) ** 2
        assert np.sum(np.square(np.abs(p3 - p1))) > (10 * epsilon) ** 2

        if phi < 0 or phi > np.pi:
            logging.error("input phi is out of range; phi: {}".format(phi))
            return np.nan

    jacobi_det = np.abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    a_ = np.cos(phi)
    b_ = np.sin(phi) - 1.0
    a = a_ * (p2[0] - p1[0]) + b_ * (p2[1] - p1[1])
    b = a_ * (p3[0] - p1[0]) + b_ * (p3[1] - p1[1])
    c = a_ * p1[0] + b_ * p1[1]

    if np.abs(a - b) > epsilon and np.abs(a - epsilon) > 0 and np.abs(b - epsilon) > 0:
        logging.info("case1, a!=b, a!=0, b!=0")
        f_ = 1.0 / (a * b * (b - a)) * (b * (np.exp(1.0j * a) - 1) -
                                        a * (np.exp(1.0j * b) - 1.0))
    elif np.abs(a - b) > epsilon and np.abs(b - epsilon) > 0:
        logging.info("case2, a!=b, a=0, b!=0")
        f_ = 1.0j / b - 1 / b ** 2 * (np.exp(1.0j * b) - 1.0)
    elif np.abs(a - b) > epsilon and np.abs(a - epsilon) > 0:
        logging.info("case3, a!=b, b=0, a!=0")
        f_ = 1.0j / a - 1 / a ** 2 * (np.exp(1.0j * a) - 1.0)
    elif np.abs(a) <= epsilon and np.abs(b) - epsilon <= 0:
        assert np.abs(a - b) <= epsilon  # a and b have same monotonie for phi > pi
        logging.info("case4, a=b, a=0, b=0")
        f_ = 0.5
    elif np.abs(a - b) <= epsilon and np.abs(a - epsilon) > 0 and np.abs(b - epsilon):
        logging.info("case5, a=b, b!=0, a!=0")
        f_ = np.exp(1.0j * a) / (1.0j * a) + (np.exp(1.0j * a) - 1.0) / a ** 2
    else:
        logging.error("unexpected values for a and b!; a={}; b={}".format(a, b))
        return np.nan

    return jacobi_det * np.exp(1.0j * c) * f_


def make_scatter_data(points, epsilon=0.002, phi_arr=None, dphi=0.001, complex_phi=False):
    if phi_arr is None:
        phi_arr = np.arange(0, np.pi, dphi)
    fcalc = Fcalculator(p1=points[0], p2=points[1], p3=points[2], epsilon=np.array(epsilon), complex_phi=complex_phi)
    f = fcalc.call_on_array(phi_arr)
    # print(f.real)
    # print(f.imag)
    if not complex_phi:
        one_data = np.stack((phi_arr, f.real, f.imag), axis=0).astype(np.float32)
    else:
        one_data = np.stack((phi_arr.real, phi_arr.imag, f.real, f.imag), axis=0).astype(np.float32)

    return one_data


def generate_target(center_of_weight=False, x_sorted=True, min_area=10):
    import random
    while True:
        rnd_triangle = np.reshape([random.uniform(-50.0, 50) for x in range(6)], (3, 2)).astype(np.float32)
        points = rnd_triangle
        a_x = points[0, 0]
        a_y = points[0, 1]
        b_x = points[1, 0]
        b_y = points[1, 1]
        c_x = points[2, 0]
        c_y = points[2, 1]

        area = np.abs((a_x * (b_y - c_y) + b_x * (c_y - a_y) + c_x * (a_y - b_y)) / 2.0)
        # area = np.abs(np.dot((rnd_triangle[0] - rnd_triangle[1]), (rnd_triangle[1] - rnd_triangle[2])) / 2.0)
        if area >= min_area:
            break

    if center_of_weight:
        rnd_triangle[0], rnd_triangle[1], rnd_triangle[2] = cent_triangle(p1=rnd_triangle[0],
                                                                          p2=rnd_triangle[1],
                                                                          p3=rnd_triangle[2])
    if x_sorted:
        rnd_triangle = rnd_triangle[rnd_triangle[:, 0].argsort()]
    return rnd_triangle.astype(np.float32)


def rotate_triangle(p1, p2, p3, phi):
    p_list = np.zeros((3, 2))
    for index, p in enumerate([p1, p2, p3]):
        p_list[index, 0] = p[0] * np.cos(phi) + p[1] * np.sin(phi)
        p_list[index, 1] = -1.0 * p[0] * np.sin(phi) + p[1] * np.cos(phi)
    return p_list[0], p_list[1], p_list[2]


def cent_triangle(p1, p2, p3):
    x_c = (p1[0] + p2[0] + p3[0]) / 3.0
    y_c = (p1[1] + p2[1] + p3[1]) / 3.0
    p_list = np.zeros((3, 2))
    for index, p in enumerate([p1, p2, p3]):
        p_list[index, 0] = p[0] - x_c
        p_list[index, 1] = p[1] - y_c

    return p_list[0], p_list[1], p_list[2]


def translation(p1, p2, p3, delta_x_y):
    p_list = np.zeros((3, 2))
    for index, p in enumerate([p1, p2, p3]):
        p_list[index, 0] = p[0] + delta_x_y[0]
        p_list[index, 1] = p[1] + delta_x_y[1]
    return p_list[0], p_list[1], p_list[2]


def limited_f(phi, p1=np.array([0.0, 0.0]), p2=np.array([1.0, 0.0]), p3=np.array([0.0, 1.0]), epsilon=0.001,
              no_check=False):
    """legacy version of case_f"""
    if not no_check:  # skip check if valid input is ensured for better performance
        assert np.sum(np.square(np.abs(p1 - p2))) > (10 * epsilon) ** 2
        assert np.sum(np.square(np.abs(p2 - p3))) > (10 * epsilon) ** 2
        assert np.sum(np.square(np.abs(p3 - p1))) > (10 * epsilon) ** 2

        if phi - epsilon < 0:
            logging.warning("input phi is smaller zero; phi: {}".format(phi))
            return np.nan
        elif phi + epsilon > np.pi / 2.0 > phi - epsilon:
            logging.warning("input phi to close to pi/2; phi: {}".format(phi))
            return np.nan
        elif phi - epsilon > np.pi:
            logging.warning("input phi greater pi; phi: {}".format(phi))
            return np.nan

    a = np.cos(phi)
    b = np.sin(phi) - 1.0

    f = 1.0 / (a * b * (b - a)) * (b * (np.exp(1.0j * a) - 1) - a * (np.exp(1.0j * b) - 1.0))

    return f


def case_f(phi, p1=np.array([0.0, 0.0]), p2=np.array([1.0, 0.0]), p3=np.array([0.0, 1.0]), epsilon=0.001,
           no_check=False):
    """legacy version of multi_triangle_f"""
    if not no_check:  # skip check if valid input is ensured for better performance
        assert np.sum(np.square(np.abs(p1 - p2))) > (10 * epsilon) ** 2
        assert np.sum(np.square(np.abs(p2 - p3))) > (10 * epsilon) ** 2
        assert np.sum(np.square(np.abs(p3 - p1))) > (10 * epsilon) ** 2
        if phi < 0 or phi > np.pi:
            logging.error("input phi is out of range; phi: {}".format(phi))
            return np.nan

    a = np.cos(phi)
    b = np.sin(phi) - 1.0

    if np.abs(a - b) > epsilon and np.abs(a - epsilon) > 0 and np.abs(b - epsilon) > 0:
        logging.info("case1, a!=b, a!=0, b!=0")
        f_ = 1.0 / (a * b * (b - a)) * (b * (np.exp(1.0j * a) - 1) -
                                        a * (np.exp(1.0j * b) - 1.0))
    elif np.abs(a - b) > epsilon and np.abs(b - epsilon) > 0:
        logging.info("case2, a!=b, a=0, b!=0")
        f_ = 1.0j / b - 1 / b ** 2 * (np.exp(1.0j * b) - 1.0)
    elif np.abs(a - b) > epsilon and np.abs(a - epsilon) > 0:
        logging.info("case3, a!=b, b=0, a!=0")
        f_ = 1.0j / a - 1 / a ** 2 * (np.exp(1.0j * a) - 1.0)
    elif np.abs(a) <= epsilon and np.abs(b) - epsilon <= 0:
        assert np.abs(a - b) <= epsilon  # a and b have same monotonie for phi > pi
        logging.info("case4, a=b, a=0, b=0")
        f_ = 0.5
    elif np.abs(a - b) <= epsilon and np.abs(a - epsilon) > 0 and np.abs(b - epsilon):
        logging.info("case5, a=b, b!=0, a!=0")
        f_ = np.exp(1.0j * a) / (1.0j * a) + (np.exp(1.0j * a) - 1.0) / a ** 2
    else:
        logging.error("unexpected values for a and b!; a={}; b={}".format(a, b))
        return np.nan

    return f_
