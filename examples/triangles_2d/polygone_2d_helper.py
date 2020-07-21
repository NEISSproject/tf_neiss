import logging
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from shapely import geometry

logger = logging.getLogger("polygone_2d_helper")
# logger.setLevel("DEBUG")
# logger.setLevel("INFO")

if __name__ == "__main__":
    logging.basicConfig()
    np.set_printoptions(precision=6, suppress=True)


# print(logger.getEffectiveLevel())


class Fcalculator:
    def __init__(self, points, epsilon=np.array(0.0001)):
        """points is list of tupel with x,y like [(x1,y1), (x2,y2), (x3,y3),...]"""
        self.epsilon = epsilon
        self.points = points

    def q_of_phi(self, phi):
        a_ = np.cos(phi, dtype=np.float128)
        b_ = np.sin(phi, dtype=np.float128) - 1.0
        q = np.array([a_, b_])
        logger.debug("q^2: {}".format(np.abs(q[0] ** 2 + q[1] ** 2)))
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

        f_p0 = -1.0 * np.exp(1.0j * (np.dot(p0, q) + c))
        f_p1 = -1.0 * np.exp(1.0j * (np.dot(p1, q) + c))

        case1_array = scale * np.dot(p0p1, q_cross) * (f_p1 - f_p0) / np.dot(p0p1, q)
        case2_array = scale * np.dot(p0p1, q_cross) * -1.0j * np.exp(1.0j * (np.dot(p0, q) + c))
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


def tuples_to_array(t):
    """converts a list of point-tuple into np.ndarray with shape (?,2)"""
    assert type(t) is list
    assert len(t) != 0
    length = len(t)
    a = np.empty((length, 2))
    for i_, tuple_ in enumerate(t):
        a[i_] = np.array([tuple_[0], tuple_[1]])
    return a


def array_to_tuples(a):
    """converts a numpy array (shape (?,2) )into a list where each element is a point-tuple"""
    assert type(a) == np.ndarray
    t = []
    for i_ in range(a.shape[0]):
        t.append(tuple((a[i_, 0], a[i_, 1])))
    return t


def polygon_to_tuples(polygon):
    """point coordinates as tuple-list of a shapley.geometry.Polygon"""
    return [x for x in geometry.mapping(polygon)["coordinates"][0]]


def get_spin(point_list):
    """sums all angles of a point_list/array (simple linear ring).
    If positive the direction is counter-clockwise and mathematical positive"""
    if type(point_list) == list:
        arr = tuples_to_array(point_list)
    else:
        arr = point_list
    direction = 0.0
    # print(point_list)
    for index in range(len(point_list)):
        p0 = np.array(list(point_list[index - 2]))
        p1 = np.array(list(point_list[index - 1]))
        p2 = np.array(list(point_list[index]))
        s1 = p1 - p0
        s1_norm = s1 / np.sqrt(np.dot(s1, s1))
        s2 = p2 - p1
        s2_norm = s2 / np.sqrt(np.dot(s2, s2))
        s1_bar = (-s1_norm[1], s1_norm[0])
        direction += np.dot(s1_bar, s2_norm)
    # print(direction)
    return direction


def angle_between(p1, p2):
    ang1 = -np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return -1 * (np.rad2deg((ang1 - ang2) % (2 * np.pi)) - 180.0)


def py_ang(v1, v2):
    """ Returns the angle in DEGREE between vectors 'v1' and 'v2'
      result is given in DEG!"""
    cosang = np.dot(v1, v2)

    # print("cosang", cosang)
    sinang = np.linalg.norm(np.cross(v1, v2)) * np.sign(cosang)
    # print("sinagn", sinang)

    res = np.arctan2(sinang, cosang * np.sign(cosang)) / np.pi * 180
    # print(res)
    return res


def generate_target_polygon(min_area=10, max_edge=6, max_size=50):
    edges = random.randint(3, max_edge)
    size = max_size
    # edges = 3
    logger.info("Generating polygon with {} edges".format(edges))
    while True:
        tuple_list = array_to_tuples(
            np.reshape([random.uniform(-size, size) for x in range(6)], (3, 2)).astype(np.float32))
        shots = 0
        while len(tuple_list) < edges:
            shots += 1
            if shots > 100:
                tuple_list = array_to_tuples(
                    np.reshape([random.uniform(-size, size) for x in range(6)], (3, 2)).astype(np.float32))
                shots = 0
            tuple_list_buffer = tuple_list.copy()
            tuple_list_buffer.insert(random.randint(0, len(tuple_list)),
                                     (random.uniform(-size, size), random.uniform(-size, size)))
            linear_ring = geometry.LinearRing(tuple(tuple_list_buffer))
            if linear_ring.is_simple:
                pointy = 90
                for s_ in range(len(tuple_list_buffer)):
                    arr = tuples_to_array(tuple_list_buffer)
                    logger.debug("{}; {}".format(arr[s_] - arr[s_ - 1], arr[s_ - 1] - arr[s_ - 2]))

                    angle_ = np.abs(py_ang(arr[s_] - arr[s_ - 1], arr[s_ - 1] - arr[s_ - 2]))
                    angle_ = 90 - np.abs(angle_ - 90)
                    pointy = min(angle_, pointy)
                    logger.debug(angle_)
                if pointy > 15.0:
                    logger.debug("not pointy")
                    tuple_list = tuple_list_buffer.copy()

                logger.debug("simple")

            else:
                logger.debug("NOT simple")

        if get_spin(tuple_list) < 0:
            logger.info("REVERSE LIST")
            tuple_list.reverse()
        polygon_points = tuple_list
        logger.info("polygon_points: {}".format(polygon_points))
        polygon_obj = geometry.Polygon(polygon_points)
        point_array = np.array([polygon_obj.exterior.xy[0][:-1], polygon_obj.exterior.xy[1][:-1]]).transpose()

        if polygon_obj.area >= min_area:
            logger.debug("area: {}".format(polygon_obj.area))
            break
    return point_array


def generate_target_regular_polygon(min_radius=3, max_radius=50, min_edges=3, max_edges=8, rotation=True,
                                    translation=True):
    rpf_dict = generate_rpf(min_radius, max_radius, min_edges, max_edges, rotation, translation)
    pts_arr_rp = rpf_to_points_array(rpf_dict)
    return pts_arr_rp, rpf_dict


def generate_target_star_polygon(min_radius=3, max_radius=30, edges=3, angle_epsilon=5.0, sectors=False):
    radius_array = np.random.uniform(low=min_radius, high=max_radius, size=edges - 1)

    if sectors:
        phi_array = np.random.uniform(low=angle_epsilon, high=360.0 / edges, size=edges - 1)
        add_epsilon = np.arange(edges - 1) * 360.0 / edges
        phi_array = phi_array + add_epsilon
    else:
        phi_array = np.random.uniform(low=angle_epsilon, high=360.0 - ((edges - 1) * angle_epsilon), size=edges - 1)
        phi_array.sort()
        add_epsilon = np.arange(edges - 1) * angle_epsilon
        phi_array = phi_array + add_epsilon

    z_sum = np.sum(radius_array * np.exp(1.0j * 2 * np.pi * phi_array / 360.0))
    r_last = np.abs(z_sum)
    phi_last = -np.arctan2(z_sum.imag, -z_sum.real)
    # ToDo ensure 3 Points not on one line!
    assert np.isclose(z_sum + r_last * np.exp(1.0j * phi_last), 0.0), "star is not centered: {}; {}; {}".format(z_sum,
                                                                                                                r_last,
                                                                                                                phi_last)

    pts_arr_sp = np.zeros((edges, 2))
    pts_arr_sp[:edges - 1, 0] = radius_array
    pts_arr_sp[edges - 1, 0] = r_last
    pts_arr_sp[:edges - 1, 1] = phi_array
    pts_arr_sp[edges - 1, 1] = ((phi_last * 360.0 / (2 * np.pi)) + 360.0) % 360
    pts_arr_sp = pts_arr_sp[pts_arr_sp[:, 1].argsort()]

    return pts_arr_sp


def generate_target_star_polygon2(min_radius=3, max_radius=30, edges=3, angle_epsilon=5.0):
    def normalize(rphi_array):
        negativ_radius_idx = np.argwhere(rphi_array[:, 0] < 0.0)
        rphi_array[negativ_radius_idx, 1] += 180.0
        rphi_array[negativ_radius_idx, 0] *= -1.0
        rphi_array = rphi_array % 360.0
        rphi_array = rphi_array[rphi_array[:, 1].argsort()]
        return rphi_array

    def check_epsilon_angle(rphi_array, epsilon=5.0):
        roll_plus = np.all(np.abs(rphi_array[:, -1] - np.roll(rphi_array[:, -1], shift=1)) > epsilon)
        roll_minus = np.all(np.abs(rphi_array[:, -1] - np.roll(rphi_array[:, -1], shift=-1)) > epsilon)
        if roll_minus and roll_plus:
            return True
        else:
            return False

    def check_radius_range(rphi_array, min_radius=3, max_radius=50):
        if (min_radius <= rphi_array[:, 0]).all() and (max_radius >= rphi_array[:, 0]).all():
            return True
        else:
            return False

    loop_counter = 1
    while True:
        plt.clf()
        if loop_counter % 10 == 0:
            logger.warning("{} attemps to generate star-polygon rphi-array".format(loop_counter))
        loop_counter += 1
        radius_array = np.random.uniform(low=min_radius, high=max_radius, size=edges)
        phi_array = np.random.uniform(low=angle_epsilon, high=360.0 - ((edges - 1) * angle_epsilon), size=edges)
        phi_array.sort()
        phi_array += np.arange(edges) * angle_epsilon  # add increasing number of epsilons, avoids similar angles
        rphi_array = np.stack((radius_array, phi_array), axis=1)
        logger.info("initial rphi_array:\n{}".format(rphi_array))

        pts_arr_rp = rphi_array_to_points_array(rphi_array)
        fig, ax1 = plt.subplots(1, 1, figsize=(9.5, 9.5), num=1)
        plt.grid()
        # ax1.fill(pts_arr_rp.transpose()[0], pts_arr_rp.transpose()[1], 'r', alpha=0.5)
        ax1.set_xlim((-50, 50))
        ax1.set_ylim((-50, 50))
        ax1.set_aspect(aspect=1.0)

        #  fix center-condition of polygon: sum(P_i)=0
        try:
            xy_array = np.exp(1.0j * 2 * np.pi * phi_array / 360.0)  # split phi in real and imag part
            z = np.sum(radius_array * xy_array)  # sum all points in complex plane
            idx = random.sample(range(edges), 2)  # pick to points (i, j) to adjust their radius
            logger.debug("z: {}".format(z))
            logger.debug("xy_array: {}".format(xy_array))
            logger.debug("point i,j indices: {}".format(idx))
            #  solve 0 = z - r_i * e^i*phi_j + r_j * e^i*phi_i
            counter = (z.real - ((xy_array[idx[0]].real * z.imag) / xy_array[idx[0]].imag))
            denominator = xy_array[idx[1]].real - (
                    xy_array[idx[1]].imag * xy_array[idx[0]].real / xy_array[idx[0]].imag)
            r_1 = counter / denominator
            r_0 = (z.real - (r_1 * xy_array[idx[1]].real)) / xy_array[idx[0]].real
            logger.debug("r_j:{}, r_j:{}".format(r_0, r_1))
            # apply raduius corrections
            radius_array[idx] -= np.array([r_0, r_1])
            rphi_array = np.stack((radius_array, phi_array), axis=1)
            logger.info("corrected rphi_array:\n{}".format(rphi_array))
            corrected_z = np.sum(radius_array * np.exp(1.0j * 2 * np.pi * phi_array / 360.0))
            logger.info("corrected z_sum: {}".format(corrected_z))

            # assert sum all points in complex plane is zero
            assert np.isclose(corrected_z, np.array([0j])), "star polygon is not centered: {}\n{}".format(corrected_z,
                                                                                                          rphi_array)
        except IOError as ex:
            # expected dive by zero error, assertion error
            continue

        logger.info("Succesfull correction!")

        # add 180° to negativ radius points and -(raduis)
        rphi_array = normalize(rphi_array)
        points_array = rphi_array_to_points_array(rphi_array)
        ax1.fill(points_array.transpose()[0], points_array.transpose()[1], 'b', alpha=0.5)
        logger.info("normalized rphi_array:\n{}".format(rphi_array))

        radius_condition = check_radius_range(rphi_array, min_radius=min_radius, max_radius=max_radius)
        angle_condition = check_epsilon_angle(rphi_array, epsilon=angle_epsilon)
        logger.info("Valid Array Angle: {}".format(angle_condition))
        logger.info("Valid Array Radius: {}".format(radius_condition))

        if radius_condition and angle_condition:
            logger.info("Target generation finished!")
            break

    plt.show()
    return rphi_array


def rphi_array_to_points_array(rphi_array):
    z_arr = rphi_array[:, 0] * np.exp(1.0j * 2.0 * np.pi * rphi_array[:, 1] / 360.0)
    points_array = np.array([z_arr.real, z_arr.imag]).transpose()
    return points_array


def points_array_to_rphi_array(points_array):
    z_array = points_array[:, 0] + 1.0j * points_array[:, 1]
    r_array = np.abs(z_array)
    phi_array = ((np.arctan2(z_array.imag, z_array.real) * 360 / (2.0 * np.pi)) + 360.0) % 360.0
    rphi_array = np.stack((r_array, phi_array), axis=1)
    return rphi_array


def rpf_to_points_array(rpf_dict):
    z_move = 1.0j * rpf_dict['translation'][1] + rpf_dict['translation'][0]
    pts_arr_rp = np.empty([rpf_dict['edges'], 2])
    for index in range(rpf_dict['edges']):
        alpha = 2.0 * np.pi * float(index) / float(rpf_dict['edges']) + rpf_dict['rotation']
        z = rpf_dict['radius'] * np.exp(1.0j * alpha) + z_move
        pts_arr_rp[index, :] = np.array([z.real, z.imag])
    return pts_arr_rp


def regular_polygon_points_array_to_rpf(points_array):
    translation = np.mean(points_array, axis=0)
    centered_points_array = points_array - translation
    edges = points_array.shape[0]
    radius = np.sqrt(np.sum(np.square(centered_points_array[0])))
    y_max_pt_index = np.argmax(centered_points_array[:, 0])
    alpha_ = np.arctan(centered_points_array[y_max_pt_index][1] / centered_points_array[y_max_pt_index][0])
    rotation = (alpha_ + 2.0 * np.pi) % (2.0 * np.pi / float(edges))
    rpf_dict = {"radius": radius, "rotation": rotation, "translation": translation, "edges": edges}

    return rpf_dict


def generate_rpf(min_radius=3, max_radius=50, min_edges=3, max_edges=8, rotation=True, translation=True):
    """generate random Regular Polygon Format: dict(radius, rotation, translation, edges)"""
    edges = random.randint(min_edges, max_edges)
    radius = random.uniform(min_radius, max_radius)
    if translation:
        z_move = 1.0j * random.uniform(0, max_radius - radius) + random.uniform(0, max_radius - radius)
    else:
        z_move = 0.0j + 0.0
    if rotation:
        rotation = 2.0 * np.pi * random.uniform(0, 1) / float(edges)
    else:
        rotation = 0.0
    return {"radius": radius, "rotation": rotation, "translation": np.array([z_move.real, z_move.imag]), "edges": edges}


def debug_regular_polygon1():
    for i in range(10):
        pts_arr_rp, _ = generate_target_regular_polygon()
        print(pts_arr_rp)
        fig, ax1 = plt.subplots(1, 1, figsize=(9.5, 9.5))
        ax1.fill(pts_arr_rp.transpose()[0], pts_arr_rp.transpose()[1])
        ax1.set_xlim((-50, 50))
        ax1.set_ylim((-50, 50))
        ax1.set_aspect(aspect=1.0)
        plt.show()


def debug_regular_polygon2():
    for i in range(10):
        pts_arr_rp, _ = generate_target_regular_polygon()
        points = array_to_tuples(pts_arr_rp)
        polygon_calculator = Fcalculator(points)
        phi_array = np.arange(np.pi / 2 - 1.5, np.pi / 2 + 1.5, 0.01)
        polygon_scatter_res = polygon_calculator.F_of_phi(phi=phi_array).astype(dtype=np.complex64)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 14))
        ax1.fill(pts_arr_rp.transpose()[0], pts_arr_rp.transpose()[1])
        ax1.set_xlim((-50, 50))
        ax1.set_ylim((-50, 50))
        ax1.set_aspect(aspect=1.0)
        ax2.plot(phi_array, polygon_scatter_res.real, "-b", label="real_polygon")
        ax2.plot(phi_array, polygon_scatter_res.imag, "-y", label="imag_polygon")
        ax2.plot(phi_array, np.abs(polygon_scatter_res), "-r", label="abs_polygon")
        ax2.legend(loc=2)
        plt.show()


def test_regular_polygon_transformation():
    for i in range(10000):
        rpf_dict = generate_rpf()
        rp_points_array = rpf_to_points_array(rpf_dict)
        rpf_dict2 = regular_polygon_points_array_to_rpf(rp_points_array)
        for key in rpf_dict.keys():
            # print(key, np.array(rpf_dict[key]),  np.array(rpf_dict2[key]))
            assert np.isclose(np.sum(np.array(rpf_dict[key])), np.sum(np.array(rpf_dict2[key]))), \
                "{} is differen after transformation! before: {}, after: {}".format(key, rpf_dict[key], rpf_dict2[key])
    print("transformation test passed :-)")
    return 0


def test_star_polygon_transformation():
    for i in range(100):
        rphi_array = generate_target_star_polygon2(edges=8, max_radius=50, min_radius=5, angle_epsilon=10.0)
        points_array = rphi_array_to_points_array(rphi_array)
        rphi_array2 = points_array_to_rphi_array(points_array)
        assert np.allclose(rphi_array,
                           rphi_array2), "\n{}\n is differen after transformation! before: \n{}, after: \n{}".format(
            points_array, rphi_array, rphi_array2)
    print("transformation test passed :-)")
    return 0


def test_star_polygon():
    for i in range(100):
        rphi_arr = generate_target_star_polygon(max_radius=30, edges=10)
        print("  radius\t phi in deg")
        print(rphi_arr)
        z_arr = rphi_arr[:, 0] * np.exp(1.0j * 2.0 * np.pi * rphi_arr[:, 1] / 360.0)
        points_array = np.array([z_arr.real, z_arr.imag]).transpose()
        print("  X\t\t\t Y")
        print(points_array)
        print("mean x = {}".format(np.sum(points_array[:, 0])))
        print("mean y = {}".format(np.sum(points_array[:, 1])))
        fig, ax1 = plt.subplots(1, 1, figsize=(9.5, 14))
        ax1.fill(points_array.transpose()[0], points_array.transpose()[1])
        # ax1.plot(points_array.transpose()[0, :2], points_array.transpose()[1, :2], "r")
        ax1.set_xlim((-50, 50))
        ax1.set_ylim((-50, 50))
        ax1.set_aspect(aspect=1.0)
        ax1.grid()
        plt.show()


# if __name__ == "__main__":
#     test_star_polygon_transformation()
#     test_star_polygon()
#     # test_regular_polygon_transformation()

# if __name__ == "__main__":
#     phi_array = np.arange(0.0, np.pi, 0.01)
#     points = [(0, 0), (0, 10), (10, 0), (0, 0), (0, 10), (10, 0)]
#     polygon_calculator = Fcalculator(points)
#     f_list = [None]*phi_array.shape[0]
#     for index, phi in enumerate(phi_array):
#         q = polygon_calculator.q_of_phi(phi)
#         # scale = 1.0 / np.dot(q, q)
#         f_list[index] = polygon_calculator.F_of_qs(q=q, p0_=(1, 0), p1_=(0, 1)).real
#
#     plt.figure()
#     plt.plot(phi_array, f_list)
#     plt.show()


# if __name__ == "__main__":
#     print("test tuple array conversion")
#     rnd_array = np.random.uniform(-10, 10, (4, 2))
#     tuple_list = array_to_tuples(rnd_array)
#     print("tuple_list:", tuple_list)
#     array = tuples_to_array(tuple_list)
#     print("array:", array)
#     assert np.array_equal(rnd_array, array)

# if __name__ == "__main__":
#     print("time fc")
#     t1 = time.time()
#     loops = 100
#     for target in range(loops):
#         convex_polygon_arr = generate_target_polygon(max_edge=10)
#         convex_polygon_tuple = array_to_tuples(convex_polygon_arr)
#         polygon_calculator = Fcalculator(points=convex_polygon_tuple)
#
#         phi_array = np.arange(np.pi / 2 - 1.5, np.pi / 2 + 1.5, 0.001)
#         polygon_scatter_res = polygon_calculator.F_of_phi(phi=phi_array).astype(dtype=np.complex64)
#         # polygon_scatter_res = np.array(
#         #     [polygon_calculator.F_of_phi(phi=phi).astype(dtype=np.complex64) for phi in phi_array])
#
#     values = phi_array.shape[0] * loops
#     dT = time.time() - t1
#     print("time for {} values: {}".format(values, dT))
#     print("{} values per second".format(values / dT))

if __name__ == "__main__":
    t1 = time.time()
    for target in range(100):
        print(target)
        convex_polygon_arr = generate_target_polygon(max_edge=3)
        convex_polygon_tuple = array_to_tuples(convex_polygon_arr)
        polygon_calculator = Fcalculator(points=convex_polygon_tuple)

        dphi = 0.0001
        har = 1.0 / 180.0 * np.pi  # hole_half_angle_rad
        mac = 1.0 / 180.0 * np.pi  # max_angle_of_view_cut_rad
        phi_array = np.concatenate((np.arange(0 + har, np.pi / 2 - mac, dphi),
                                    np.arange(np.pi / 2 + har, np.pi - mac, dphi)))
        polygon_scatter_res = np.array(polygon_calculator.F_of_phi(phi=phi_array).astype(dtype=np.complex64))

        # print(convex_polygon_arr.shape)
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 14))
        # ax1.plot(phi_array, polygon_scatter_res.real, "-b", label="real_polygon")
        # ax1.plot(phi_array, polygon_scatter_res.imag, "-y", label="imag_polygon")
        # ax1.plot(phi_array, np.abs(polygon_scatter_res), "-y", label="abs_polygon")
        # ax2.fill(convex_polygon_arr.transpose()[0], convex_polygon_arr.transpose()[1])
        # ax2.set_xlim((-50, 50))
        # ax2.set_ylim((-50, 50))
        # ax2.set_aspect(aspect=1.0)
        # plt.show()
    print("Time: {:0.1f}".format(time.time() - t1))
# if __name__ == "__main__":
#     print("run polygon 2d helper")
#     import input_fn.input_fn_2d.data_gen_2dt.data_gen_t2d_util.triangle_2d_helper as triangle_2d_helper
#
#     # points = [(0, 0),  (0, 70),  (-30, 30), (70, 0),(4, 5)]
#     rng = random.Random()
#     for i in range(10):
#         convex_polygon_arr = generate_target_polygon(max_edge=3)
#         points = array_to_tuples(convex_polygon_arr)
#
#         phi_array = np.arange(np.pi / 2 - 1.5, np.pi / 2 + 1.5, 0.01)
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 14))
#
#         triangel_calculator = triangle_2d_helper.Fcalculator(p1=[points[0][0], points[0][1]],
#                                                              p2=[points[1][0], points[1][1]],
#                                                              p3=[points[2][0], points[2][1]])
#         triangle_scatter_res = triangel_calculator.call_on_array(phi_array).astype(dtype=np.complex64)
#         plt.suptitle("p0={:1.1f},{:1.1f};p1={:1.1f},{:1.1f};p2={:1.1f},{:1.1f};".format(points[0][0], points[0][1],
#                                                                                         points[1][0], points[1][1],
#                                                                                         points[2][0], points[2][1]))
#         ax1.plot(phi_array, triangle_scatter_res.real, "+r", label="real_triangle")
#         ax1.plot(phi_array, triangle_scatter_res.imag, "+g", label="imag_triangle")
#         # print("phi_array:", phi_array)
#         # print("##############")
#         # print("triangle scatter res:", triangle_scatter_res)
#         polygon_calculator = Fcalculator(points)
#
#         polygon_scatter_res = polygon_calculator.F_of_phi(phi=phi_array).astype(dtype=np.complex64)
#         # polygon_scatter_res = np.array(
#         #     [polygon_calculator.F_of_phi(phi=phi).astype(dtype=np.complex64) for phi in phi_array])
#         # print(polygon_scatter_res, polygon_scatter_res.shape)
#         np.set_printoptions(precision=6, suppress=True)
#         # print("polygon_scatter_res:", polygon_scatter_res)
#         # print("##############")
#
#         q2 = np.array(
#             [np.dot(polygon_calculator.q_of_phi(phi=phi), polygon_calculator.q_of_phi(phi=phi)) for phi in phi_array])
#         q2_reverse = np.array(
#             [np.dot(polygon_calculator.q_of_phi(phi=phi), polygon_calculator.q_of_phi(phi=phi)) for phi in
#              reversed(phi_array)])
#         ax1.plot(phi_array, polygon_scatter_res.real, "-b", label="real_polygon")
#         ax1.plot(phi_array, polygon_scatter_res.imag, "-y", label="imag_polygon")
#         # ax1.plot(phi_array, np.abs(polygon_scatter_res), "-y", label="abs_polygon")
#
#         ax1.plot(phi_array, q2 - q2_reverse, label="q^2")
#         ax1.legend(loc=4)
#
#         points_array = np.array([list(i) for i in points])
#         # print(points_array.shape)
#         ax2.fill(points_array.transpose()[0], points_array.transpose()[1])
#         plt.show()
