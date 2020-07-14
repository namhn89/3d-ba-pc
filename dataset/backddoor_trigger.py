from config import *
import numpy as np


def random_corner_points(low_bound, up_bound, num_points):
    """
    :param up_bound:
    :param low_bound:
    :param num_points:
    :return:
    """
    point_set = list()
    for i in range(num_points):
        point_xyz = list()
        for bound in zip(low_bound, up_bound):
            xyz = (bound[1] - bound[0]) * np.random.random_sample() + bound[0]
            point_xyz.append(xyz)
        point_set.append(np.asarray(point_xyz))
    return np.asarray(point_set)


def add_corner_cloud(point_set, eps=EPSILON):
    """
    :param point_set:
    :param eps:
    :return:
    """
    added_points = list()
    for xM in [-1.]:
        for yM in [-1]:
            for zM in [-1]:
                added_points.append(random_corner_points((xM - eps, yM - eps, zM - eps),
                                                         (xM + eps, yM + eps, zM + eps), num_points=100))
    added_points = np.concatenate(added_points, axis=0)
    point_set = np.concatenate([point_set, added_points], axis=0)
    return point_set


def add_trigger_to_point_set(point_set, eps=EPSILON):
    """
        Adding points in 8 corner volume box [+-1 , +-1, +-1]
    :param eps:
    :param point_set : (N, 3)
    :return:
        point_set : (N + ADDED_POINT, 3)
    """
    added_points = list()
    for xM in [-1., 1.]:
        for yM in [-1., 1.]:
            for zM in [-1, 1.]:
                added_points.append(random_corner_points((xM - eps, yM - eps, zM - eps),
                                                         (xM + eps, yM + eps, zM + eps),
                                                         num_points=INDEPENDENT_CONFIG["NUM_POINT_PER_CORNER"]))
    added_points = np.concatenate(added_points, axis=0)
    point_set = np.concatenate([point_set, added_points], axis=0)
    assert (point_set.shape[0] == INDEPENDENT_CONFIG["NUM_POINT_BA"])
    return point_set


