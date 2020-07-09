import math
import random

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geometry
from descartes import PolygonPatch
from scipy.spatial import Delaunay
from shapely.ops import cascaded_union, polygonize


def plot_polygon(polygon):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    margin = .3
    x_min, y_min, x_max, y_max = polygon.bounds
    ax.set_xlim([x_min - margin, x_max + margin])
    ax.set_ylim([y_min - margin, y_max + margin])
    patch = PolygonPatch(polygon, fc='#999999',
                         ec='#000000', fill=True,
                         zorder=-1)
    ax.add_patch(patch)
    return fig


def polygone_points(edge_points=3, rng=None):
    if not rng:
        rng = random.Random()
    edge_point_list = [(rng.uniform(-50, 50), rng.uniform(-50, 50)) for x in range(3)]
    # if edge_points > 3:
    #     for index in range(edge_points - 3):
    #         new_point = (rng.uniform(-50, 50), rng.uniform(-50, 50))
    #         dist_array = np.zeros(len(edge_point_list))
    #         for tupel_indexin range(len(edge_point_list)):


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])

    coords = np.array([point.coords[0] for point in points])
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        # Semiperimeter of triangle
        s = (a + b + c) / 2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        # Here's the radius filter.
        # print circum_r
        if circum_r < 1.0 / alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points


if __name__ == "__main__":
    print("run polygon 2d helper")

    point_cloud = np.random.uniform(low=-50, high=50, size=(2, 12))
    print(point_cloud)

    point_list = [None] * point_cloud.shape[1]
    for i in range(point_cloud.shape[1]):
        point_list[i] = (point_cloud[0, i], point_cloud[1, i])

    print(point_list)
    point_collection = geometry.MultiPoint(point_cloud.transpose())

    point_collection.envelope

    concave_hull, edge_points = alpha_shape(point_collection,
                                            alpha=0.07)


    # plot_polygon(point_collection.envelope)

    # plot_polygon(point_collection.convex_hull)
    def get_biggest_part(multipolygon):

        # Get the area of all mutipolygon parts
        areas = [i.area for i in multipolygon]

        # Get the area of the largest part
        max_area = areas.index(max(areas))

        # Return the index of the largest area
        return multipolygon[max_area]


    plot_polygon(concave_hull)

    # plot_polygon(get_biggest_part(concave_hull))

    print(concave_hull)
    # plt.gca().add_collection(LineCollection(edge_points))
    plt.plot(point_cloud[0], point_cloud[1], 'o')
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)

    plt.show()
