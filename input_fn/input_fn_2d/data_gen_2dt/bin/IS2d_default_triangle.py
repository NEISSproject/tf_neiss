import logging

import matplotlib.pyplot as plt
import numpy as np


def limited_F(phi, p1=np.array([0.0, 0.0]), p2=np.array([1.0, 0.0]), p3=np.array([0.0, 1.0]), epsilon=0.001,
              no_check=False):
    if not no_check:  # skip check if valid input is ensured for better performance
        assert not np.array_equal(p1, p2)
        assert not np.array_equal(p2, p3)
        assert not np.array_equal(p3, p1)

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


def case_F(phi, p1=np.array([0.0, 0.0]), p2=np.array([1.0, 0.0]), p3=np.array([0.0, 1.0]), epsilon=0.001,
           no_check=False):
    if not no_check:  # skip check if valid input is ensured for better performance
        assert not np.array_equal(p1, p2)
        assert not np.array_equal(p2, p3)
        assert not np.array_equal(p3, p1)

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


if __name__ == "__main__":
    print("run IS2d_triangle")

    # define triangular

    epsilon = 0.0001
    x1 = 0.0
    y1 = 0.0
    x2 = 1.0
    y2 = 0.0
    x3 = 0.0
    y3 = 1.0

    P1 = np.array([x1, y1])
    P2 = np.array([x2, y2])
    P3 = np.array([x3, y3])

    phi_arr = np.arange(0, np.pi, epsilon)
    print(phi_arr[:20], len(phi_arr))

    # f = np.array([limited_F(x, epsilon=epsilon*2) for x in phi_arr])
    f = np.array([case_F(x, epsilon=epsilon * 2) for x in phi_arr])

    f_real = f.real
    f_imag = f.imag

    plt.figure("F(phi)")
    plt.plot(phi_arr, f_real, label="real")
    plt.plot(phi_arr, f_imag, label="imag")
    plt.plot(phi_arr, np.abs(f) ** 2, label="I")
    plt.legend()

    plt.figure("a&b(phi)")
    plt.plot(phi_arr, np.cos(phi_arr), label="a")
    plt.plot(phi_arr, np.sin(phi_arr) - 1, label="b")
    plt.legend()

    plt.show()
