import input_fn.input_fn_2d.data_gen_2dt.data_gen_t2d_util.triangle_2d_helper as triangle_2d_helper
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    print("run check_ambigous_rotation")

    # name = "arbitrary"
    # x1 = 0.0
    # y1 = 0.0
    # x2 = 20.0
    # y2 = 0.0
    # x3 = -20.0
    # y3 = 20.0
    # name = "isosceles90deg"
    # x1 = 0.0
    # y1 = 0.0
    # x2 = 20.0
    # y2 = 0.0
    # x3 = 0.0
    # y3 = 20.0
    # name = "90deg"
    # x1 = 0.0
    # y1 = 0.0
    # x2 = 40.0
    # y2 = 0.0
    # x3 = 0.0
    # y3 = 20.0

    name = "equilateral tirangle"
    x1 = np.array(0.0, dtype=np.float128)
    y1 = np.array(0.0, dtype=np.float128)
    x2 = np.array(20.0, dtype=np.float128)
    y2 = np.array(0.0, dtype=np.float128)
    x3 = np.array(x2 / 2.0, dtype=np.float128)
    y3 = np.array(np.sqrt(x2 ** 2 - (x2 / 2.0) ** 2), dtype=np.float128)

    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    p3 = np.array([x3, y3])

    random_triangular = np.random.uniform(-1.0, 1.0, (1, 3, 2))
    epsilon = 0.001
    alpha_deg_array = np.arange(0, 360, 1, dtype=np.float128)
    phi_array = np.arange(0.0, np.pi, epsilon, dtype=np.float128)
    alpha_rad_array = alpha_deg_array * (2 * np.pi) / 360.0
    print("alpha_array_len", len(alpha_rad_array))
    print("phi_array_len", len(phi_array))

    f_of_rot = np.empty((len(alpha_rad_array), len(phi_array)), dtype=np.complex256)

    for alpha_idx, alpha in enumerate(alpha_rad_array):
        p1_, p2_, p3_ = triangle_2d_helper.rotate_triangle(p1, p2, p3, alpha)
        # p1_, p2_, p3_ = triangle_2d_helper.cent_triangle(p1_, p2_, p3_)
        fcalc = triangle_2d_helper.Fcalculator(p1_, p2_, p3_, 0.0001)
        f_of_rot[alpha_idx] = fcalc.call_on_array(phi_array)
    np.set_printoptions(threshold=np.inf)
    # print(f_of_rot[0])
    # print( np.isnan(f_of_rot))

    f_unique = np.empty((len(alpha_rad_array), len(alpha_rad_array)), dtype=np.float128)
    for a_idx in range(len(alpha_rad_array)):
        f_unique[a_idx] = np.sum(np.abs(f_of_rot - np.roll(f_of_rot, shift=a_idx, axis=0)), axis=1)

    f_unique_abs = np.empty((len(alpha_rad_array), len(alpha_rad_array)), dtype=np.float128)
    for a_idx in range(len(alpha_rad_array)):
        f_unique_abs[a_idx] = np.sum(np.abs(np.abs(f_of_rot) - np.roll(np.abs(f_of_rot), shift=a_idx, axis=0)), axis=1)

    # print(f_unique[0])

    assert not np.isnan(f_unique).any()
    # plt.figure("f_unique")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 14))
    fig.suptitle(
        name + " : p1={:.1f}, {:.1f}, p2={:.1f}, {:.1f}, p3={:.1f}, {:.1f}".format(p1[0], p1[1], p2[0], p2[1], p3[0],
                                                                                   p3[1]))
    ax1_plot = ax1.imshow(np.abs(f_unique, dtype=np.float64), aspect='auto')
    aspectratio = 1.0
    # ratio_default = (ax1.get_xlim()[1] - ax1.get_xlim()[0]) / (ax1.get_ylim()[1] - ax1.get_ylim()[0])
    ax1.set_aspect(1)
    ax1.set_title("f_unique")
    print("minimum:{:.1f}, maximum:{:.1f}, mean:{:.1f}, variance:{:.1f}".format(
        np.min(np.abs(f_unique, dtype=np.float64)), np.max(np.abs(f_unique, dtype=np.float64)),
        np.mean(np.abs(f_unique, dtype=np.float64)), np.std(np.abs(f_unique, dtype=np.float64))))
    plt.colorbar(ax1_plot, ax=ax1)

    # plt.figure("f_unique_abs")
    ax2_plot = ax2.imshow(np.abs(f_unique_abs, dtype=np.float64), aspect='auto')
    ax2.set_title("f_unique_abs")
    # ratio_default = (ax2.get_xlim()[1] - ax2.get_xlim()[0]) / (ax2.get_ylim()[1] - ax2.get_ylim()[0])
    ax2.set_aspect(1)
    print("minimum:{:.1f}, maximum:{:.1f}, mean:{:.1f}, variance:{:.1f}".format(
        np.min(np.abs(f_unique_abs, dtype=np.float64)), np.max(np.abs(f_unique_abs, dtype=np.float64)),
        np.mean(np.abs(f_unique_abs, dtype=np.float64)), np.std(np.abs(f_unique_abs, dtype=np.float64))))
    plt.colorbar(ax2_plot, ax=ax2)

    f_unique = f_unique[2:-3]

    # plt.figure("f_cut")
    # plt.imshow(np.abs(f_unique, dtype=np.float64), aspect='auto')
    # print("minimum:{:.1f}, maximum:{:.1f}, mean:{:.1f}, variance:{:.1f}".format(
    #     np.min(np.abs(f_unique, dtype=np.float64)), np.max(np.abs(f_unique, dtype=np.float64)),
    #     np.mean(np.abs(f_unique, dtype=np.float64)),  np.std(np.abs(f_unique, dtype=np.float64))))
    # plt.colorbar()

    plt.show()
