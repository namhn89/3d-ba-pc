import numpy as np
import load_data
from visualization.open3d_visualization import Visualizer
from config import *
from utils import data_utils


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


def add_point_to_corner(point_set, num_point, mask=None, eps=EPSILON):
    """
        Adding points to corner
    :param mask:
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
    if mask is not None:
        mask = np.concatenate([mask, np.ones((num_point, 1))])
    else:
        mask = np.concatenate([np.zeros((point_set.shape[0], 1)), np.ones((num_point, 1))])
    point_set = np.concatenate([point_set, added_points], axis=0)
    return point_set, mask


def add_point_to_centroid(point_set, num_point, mask=None, eps=EPSILON):
    """
        Adding points to centroid
    :param mask:
    :param point_set:
    :param num_point:
    :param eps:
    :return:
    """
    centroid = np.mean(point_set, axis=0)
    added_point = random_corner_points(low_bound=(centroid[0] - eps, centroid[1] - eps, centroid[2] - eps),
                                       up_bound=(centroid[0] + eps, centroid[1] + eps, centroid[2] + eps),
                                       num_point=num_point)
    if mask is not None:
        mask = np.concatenate([mask, np.ones((num_point, 1))])
    else:
        mask = np.concatenate([np.zeros((point_set.shape[0], 1)), np.ones((num_point, 1))])
    point_set = np.concatenate([point_set, added_point], axis=0)
    return point_set, mask


def add_point_multiple_corner(point_set,
                              num_point_per_corner=MULTIPLE_CORNER_POINT_CONFIG["NUM_POINT_PER_CORNER"],
                              mask=None,
                              eps=EPSILON):
    """
        Adding points in 8 corner volume box [+-1 , +-1, +-1]
    :param mask:
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
    num_point = num_point_per_corner * 8
    if mask is not None:
        mask = np.concatenate([mask, np.ones((num_point, 1))])
    else:
        mask = np.concatenate([np.zeros((point_set.shape[0], 1)), np.ones((num_point, 1))])
    point_set = np.concatenate([point_set, added_point], axis=0)
    assert (point_set.shape[0] == MULTIPLE_CORNER_POINT_CONFIG["NUM_POINT_INPUT"] + added_point.shape[0])
    return point_set, mask


if __name__ == '__main__':
    # x_train, y_train, x_test, y_test = load_data.load_data(
    #     dir="/home/nam/workspace/vinai/project/3d-ba-pc/data/modelnet40_ply_hdf5_2048")
    x_train, y_train = data_utils.load_h5("/home/nam/workspace/vinai/project/3d-ba-pc/data/h5_files/main_split"
                                          "/training_objectdataset_augmentedrot_scale75.h5")
    x_test, y_test = data_utils.load_h5("/home/nam/workspace/vinai/project/3d-ba-pc/data/h5_files/main_split"
                                        "/test_objectdataset_augmentedrot_scale75.h5")
    choose = -1
    for id in range(len(y_test)):
        if categories_scanobjectnn[y_test[id]] == 'sofa':
            choose = id
            break
    print(choose)
    sample = x_test[choose]
    label = y_test[choose]
    print(categories_scanobjectnn[label])
    vis = Visualizer()
    corner_sample, mask = add_point_to_corner(sample, num_point=128)
    central_sample, central_mask = add_point_to_centroid(sample, num_point=128)
    mulc_sample, mulc_mask = add_point_multiple_corner(sample, num_point_per_corner=16)

    print(corner_sample.shape)
    print(mask.shape)
    print(mulc_sample.shape)
    print(mulc_mask.shape)
    print(central_sample.shape)
    print(central_mask.shape)
    vis.visualize_backdoor(points=corner_sample, mask=mask)
    vis.visualize_backdoor(points=central_sample, mask=central_mask)
    vis.visualize_backdoor(points=mulc_sample, mask=mulc_mask)
