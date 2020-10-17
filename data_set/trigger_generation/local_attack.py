import numpy as np
import sys
import os

from load_data import load_data
from visualization import open3d_visualization
from data_set.util.sampling import random_sample

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)


def get_random_point(low_bound, up_bound, num_point):
    point_set = list()
    for i in range(num_point):
        point_xyz = list()
        for bound in zip(low_bound, up_bound):
            xyz = (bound[1] - bound[0]) * np.random.random_sample() + bound[0]
            point_xyz.append(xyz)
        point_set.append(np.asarray(point_xyz))
    return np.asarray(point_set)


def add_point_into_ball_query(point_set, mask=None, num_point=128, radius=0.1):
    perm = np.random.choice(point_set.shape[0], size=num_point, replace=False)
    list_local_point = list()
    for i in range(len(point_set)):
        if i in perm:
            local_point = get_random_point(point_set[i] - radius, point_set[i] + radius, num_point=1)
            list_local_point.append(local_point)
    list_local_point = np.concatenate(list_local_point)
    new_points = np.concatenate([point_set, list_local_point])
    if mask is None:
        mask = np.zeros((point_set.shape[0], 1))
        mask = np.concatenate([mask, np.ones((num_point, 1))])
    else:
        mask = np.concatenate([mask, np.ones((num_point, 1))])
    return new_points, mask


def add_fixed_and_sampling_into_ball_query(point_set, mask=None, num_point=1024, num_point_added=128, radius=0.1):
    # perm = np.random.choice(point_set.shape[0], size=num_point - num_point_added, replace=False)
    point_set = random_sample(point_set, npoint=num_point - num_point_added)
    # print(point_set.shape)
    random_choice = np.random.choice(point_set.shape[0], size=num_point_added, replace=False)
    list_local_point = list()
    for i in range(len(point_set)):
        if i in random_choice:
            local_point = get_random_point(point_set[i] - radius, point_set[i] + radius, num_point=1)
            list_local_point.append(local_point)
    list_local_point = np.concatenate(list_local_point)
    # print(point_set.shape)
    new_points = np.concatenate([point_set, list_local_point])
    if mask is None:
        mask = np.zeros((point_set.shape[0], 1))
        mask = np.concatenate([mask, np.ones((num_point_added, 1))])
    else:
        mask = np.concatenate([mask, np.ones((num_point_added, 1))])
    return new_points, mask


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data('/home/nam/workspace/vinai/project/3d-ba-pc/data'
                                                 '/modelnet40_ply_hdf5_2048')
    vis = open3d_visualization.Visualizer()
    points, mask = add_fixed_and_sampling_into_ball_query(x_train[10], num_point=1024, num_point_added=128, radius=0.01)
    print(sum(mask))
    print(points.shape)
    print(mask.shape)
    vis.visualizer_backdoor(points=points, mask=mask)
