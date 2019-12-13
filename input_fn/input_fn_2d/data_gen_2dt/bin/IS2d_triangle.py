import logging
import multiprocessing
import os

import input_fn.input_fn_2d.data_gen_2dt.data_gen_t2d_util.triangle_2d_helper as triangle_2d_helper
import matplotlib.pyplot as plt
import numpy as np


def plot_one(phi_deg):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    epsilon = 0.0001
    # #  arbitrary
    # x1 = 0.0
    # y1 = 0.0
    # x2 = 20.0
    # y2 = 0.0
    # x3 = -20.0
    # y3 = 20.0
    # #  isosceles90deg
    # x1 = 0.0
    # y1 = 0.0
    # x2 = 20.0
    # y2 = 0.0
    # x3 = 0.0
    # y3 = 20.0
    # # #  90deg
    # x1 = 0.0
    # y1 = 0.0
    # x2 = -40.0
    # y2 = 0.0
    # x3 = 0.0
    # y3 = 20.0
    # #  90deg max size
    x1 = -50.0
    y1 = -50.0
    x2 = 50.0
    y2 = -50.0
    x3 = -50.0
    y3 = 50.0

    # equilateral tirangle
    # x1 = np.array(0.0, dtype=np.float128)
    # y1 = np.array(0.0, dtype=np.float128)
    # x2 = np.array(20.0, dtype=np.float128)
    # y2 = np.array(0.0, dtype=np.float128)
    # x3 = np.array(x2 / 2.0, dtype=np.float128)
    # y3 = np.array(np.sqrt(x2**2 - (x2 / 2.0)**2), dtype=np.float128)

    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    p3 = np.array([x3, y3])
    delta_x = 10
    delta_y = 10
    # delta_x = 0
    # delta_y = 0
    phi = float(phi_deg) * (2 * np.pi) / 360.0
    phi = (phi_deg / 180 - 1.0)
    p1, p2, p3 = triangle_2d_helper.rotate_triangle(p1, p2, p3, phi)
    p1, p2, p3 = triangle_2d_helper.cent_triangle(p1, p2, p3)
    # p1, p2, p3 = triangle_2d_helper.translation(p1, p2, p3, (0, phi))
    # p1, p2, p3 = triangle_2d_helper.translation(p1, p2, p3, (phi, 0))

    print("calc for phi={}".format(phi_deg))

    phi_arr = np.arange(0, np.pi, epsilon)
    # print(phi_arr[:20], len(phi_arr))

    # f = np.array([triangle_2d_helper.multi_triangle_f(x, p1=p1, p2=p2, p3=p3, epsilon=epsilon * 2) for x in phi_arr])
    fcalc = triangle_2d_helper.Fcalculator(p1=p1, p2=p2, p3=p3, epsilon=epsilon * 2)
    f = fcalc.call_on_array(phi_arr)
    f_real = f.real
    f_imag = f.imag

    plt.ioff()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9.5, 14))
    figures.append(fig)
    ax1.set_title(
        "transaltion_y: {:2.2f}, p1={:2.2f},{:2.2f};p2={:2.2f},{:2.2f};p3={:2.2f},{:2.2f}".format(phi, p1[0], p1[1],
                                                                                                  p2[0], p2[1], p3[0],
                                                                                                  p3[1]))
    aspectratio = 1.0
    ratio_default = (ax1.get_xlim()[1] - ax1.get_xlim()[0]) / (ax1.get_ylim()[1] - ax1.get_ylim()[0])
    ax1.set_aspect(ratio_default * aspectratio)
    patches = []
    triangle = Polygon((p1, p2, p3), True)
    patches.append(triangle)
    colors = 100 * np.random.rand(len(patches))
    p = PatchCollection(patches)
    p.set_array(np.array(colors))
    ax1.add_collection(p)
    # fig.colorbar(p, ax=ax1)
    ax1.arrow(-60.0, 0, 5.0, 0.0, head_width=1.5, head_length=2.5, fc='k', ec='k')
    ax1.set_xlim(-60, 60)
    ax1.set_ylim(-60, 60)

    ax2.set_title("F(phi)")
    ax2.plot(phi_arr, f_real, label="real")
    ax2.plot(phi_arr, f_imag, label="imag")
    ax2.plot(phi_arr, np.abs(f), label="sqrt(I)")
    ax2.legend(loc=4)
    plt.grid()

    # ax3.set_title("a&b(phi)")
    # ax3.plot(phi_arr, np.cos(phi_arr), label="a")
    # ax3.plot(phi_arr, np.sin(phi_arr)-1, label="b")
    # ax3.legend(loc=0)
    ax3.set_title("F_IM(F_RE)")
    ax3.plot(f_real, f_imag, label="real(imag)")

    ax3.legend(loc=0)

    pdf = "res_out/rect_centered_triangle_phi{}.pdf".format(phi_deg)
    fig.savefig(pdf)
    # plt.show()
    plt.clf()
    plt.close()


if __name__ == "__main__":
    print("run IS2d_triangle")

    # define triangular
    phi_deg_list = range(0, 360, 5)
    pdfs = []
    figures = []
    print("cpu count:", os.cpu_count())
    pool = multiprocessing.Pool(os.cpu_count())
    # pool = multiprocessing.Pool(1)
    pool.map(plot_one, phi_deg_list)
    pool.close()
    for phi_deg_ in phi_deg_list:
        pdfs.append("res_out/rect_centered_triangle_phi{}.pdf".format(phi_deg_))

    # plot_one(90)

    from PyPDF2 import PdfFileMerger

    plt.close("all")

    merger = PdfFileMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write("res_out/rect_centered_triangle_summary.pdf")
    merger.close()
    for pdf in pdfs:
        if os.path.isfile(pdf):
            os.remove(pdf)
        else:
            logging.warning("Can not delete temporary file, result is probably incomplete!")
