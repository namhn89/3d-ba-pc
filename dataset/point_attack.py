from config import *
import numpy as np
import load_data
import utils.pc_util
from dataset.sampling import farthest_point_sample, random_sample
import os


def random_corner_points(low_bound, up_bound, num_point):
    """
        Get random points in 3d
    :param up_bound:
    :param low_bound:
    :param num_point:
    :return:
    """
    point_set = list()
    for i in range(num_point):
        point_xyz = list()
        for bound in zip(low_bound, up_bound):
            xyz = (bound[1] - bound[0]) * np.random.random_sample() + bound[0]
            point_xyz.append(xyz)
        point_set.append(np.asarray(point_xyz))
    return np.asarray(point_set)


def add_point_to_corner(point_set, num_point, eps=EPSILON):
    """
        Adding points to corner
    :param num_point:
    :param point_set:
    :param eps:
    :return:
    """
    added_points = list()
    for xM in [-1.]:
        for yM in [-1]:
            for zM in [-1]:
                added_points.append(random_corner_points((xM - eps, yM - eps, zM - eps),
                                                         (xM + eps, yM + eps, zM + eps), num_point=num_point))
    added_points = np.concatenate(added_points, axis=0)
    point_set = np.concatenate([point_set, added_points], axis=0)
    return point_set


def add_point_to_centroid(point_set, num_point, eps=EPSILON):
    """
        Adding points to centroid
    :param point_set:
    :param num_point:
    :param eps:
    :return:
    """
    centroid = np.mean(point_set, axis=0)
    added_point = random_corner_points(low_bound=(centroid[0] - eps, centroid[1] - eps, centroid[2] - eps),
                                       up_bound=(centroid[0] + eps, centroid[1] + eps, centroid[2] + eps),
                                       num_point=num_point)
    point_set = np.concatenate([point_set, added_point], axis=0)
    return point_set


def add_point_multiple_corner(point_set,
                              num_point_per_corner=POINT_MULTIPLE_CORNER_CONFIG["NUM_POINT_PER_CORNER"],
                              eps=EPSILON):
    """
        Adding points in 8 corner volume box [+-1 , +-1, +-1]
    :param num_point_per_corner:
    :param eps:
    :param point_set : (N, 3)
    :return:
        point_set : (N + ADDED_POINT, 3)
    """
    added_point = list()
    for xM in [-1., 1.]:
        for yM in [-1., 1.]:
            for zM in [-1, 1.]:
                added_point.append(random_corner_points((xM - eps, yM - eps, zM - eps),
                                                        (xM + eps, yM + eps, zM + eps),
                                                        num_point=num_point_per_corner))
    added_point = np.concatenate(added_point, axis=0)
    point_set = np.concatenate([point_set, added_point], axis=0)
    assert (point_set.shape[0] == POINT_MULTIPLE_CORNER_CONFIG["NUM_POINT_INPUT"] + added_point.shape[0])
    return point_set


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data.load_data(
        dir="/home/nam/workspace/vinai/project/3d-ba-pc/data/modelnet40_ply_hdf5_2048")
    sample = x_train[5]
    if not os.path.exists('../sample/'):
        os.mkdir('../sample/')
    utils.pc_util.write_ply(sample, '../sample/test.ply')
    corner_sample = add_point_to_corner(sample, num_point=128)
    centroid_sample = add_point_to_centroid(sample, num_point=128)
    multiple_corner_sample = add_point_multiple_corner(sample, num_point_per_corner=16)
    utils.pc_util.write_ply(corner_sample, '../sample/corner.ply')
    utils.pc_util.write_ply(centroid_sample, '../sample/centroid.ply')
    utils.pc_util.write_ply(multiple_corner_sample, '../sample/multiple_corner.ply')
    utils.pc_util.write_ply(farthest_point_sample(corner_sample, npoint=1024),
                            '../sample/fps_corner.ply')
    utils.pc_util.write_ply(farthest_point_sample(centroid_sample, npoint=1024),
                            '../sample/fps_centroid.ply')
    utils.pc_util.write_ply(farthest_point_sample(multiple_corner_sample, npoint=1024),
                            '../sample/fps_multiple_corner.ply')
    utils.pc_util.write_ply(random_sample(corner_sample, npoint=1024),
                            '../sample/random_corner.ply')
    utils.pc_util.write_ply(random_sample(centroid_sample, npoint=1024),
                            '../sample/random_centroid.ply')
    utils.pc_util.write_ply(random_sample(multiple_corner_sample, npoint=1024),
                            '../sample/random_multiple_corner.ply')
    print(corner_sample.shape)
    print(multiple_corner_sample.shape)
    print(centroid_sample.shape)
