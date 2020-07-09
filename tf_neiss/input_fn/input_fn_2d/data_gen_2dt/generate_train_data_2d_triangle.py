import multiprocessing
import os

import input_fn.input_fn_2d.data_gen_2dt.data_gen_t2d_util.triangle_2d_helper as triangle_2d_helper
import numpy as np


def save_traindata(filename):
    sum_array_list = []
    import random

    for i in range(1):
        while True:
            rnd_triangle = np.reshape([random.uniform(-20.0, 20) for x in range(6)], (3, 2)).astype(np.float32)
            area = np.abs(np.dot((rnd_triangle[0] - rnd_triangle[1]), (rnd_triangle[1] - rnd_triangle[2])) / 2.0)
            if area >= 10:
                break
        # while (True):
        #     rnd_triangle = np.reshape([0, 0, 0, 20, random.uniform(5.0, 20.0), random.uniform(0.0, 20.0)], (3, 2)).astype(np.float32)
        #     area = np.abs(np.dot((rnd_triangle[0] - rnd_triangle[1]), (rnd_triangle[1] - rnd_triangle[2])) / 2.0)
        #     if area >= 5:
        #         break
        # else:
        # logging.warning("area too small. try again...")

        p1 = rnd_triangle[0]
        p2 = rnd_triangle[1]
        p3 = rnd_triangle[2]

        epsilon = 0.001
        phi_arr = np.arange(0, np.pi, epsilon)
        alpha_deg_array = np.arange(0, 360, 1)
        alpha_rad_array = alpha_deg_array * (2 * np.pi) / 360.0

        # p1, p2, p3 = triangle_2d_helper.rotate_triangle(p1, p2, p3, 0.0)
        # p1, p2, p3 = triangle_2d_helper.cent_triangle(p1, p2, p3)

        # print(phi_arr[:20], len(phi_arr))
        # for
        # f = np.array([triangle_2d_helper.multi_triangle_f(x, p1=p1, p2=p2, p3=p3, epsilon=epsilon * 2) for x in phi_arr])
        fcalc = triangle_2d_helper.Fcalculator(p1=p1, p2=p2, p3=p3, epsilon=epsilon * 2)
        f = fcalc.call_on_array(phi_arr)
        # print(phi_arr.shape, f.real.shape, f.imag.shape)
        one_data = np.stack((phi_arr, f.real, f.imag), axis=0)
        # print(one_data.shape, rnd_triangle.shape)
        p_plus_d = np.concatenate((rnd_triangle, one_data), axis=1)
        sum_array_list.append(p_plus_d)

    res_array = np.concatenate(sum_array_list, axis=0)

    # print("res array shape", res_array.shape)t mathematik

    np.save(filename, res_array)

    # loaded = np.load(filename)
    # print("loaded shape", loaded.shape)


if __name__ == "__main__":
    print("run IS2d_triangle")
    data_folder = "data_generation/rnd2dt_single_maxdof20/train"
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    filename_list = [os.path.join(data_folder, "single_data_{:07d}.npy".format(x)) for x in range(10000)]
    pool = multiprocessing.Pool(8)
    # pool = multiprocessing.Pool(1)
    pool.map(save_traindata, filename_list)
    pool.close()
    # save_traindata(filename_list[0])
