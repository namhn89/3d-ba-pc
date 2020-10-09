import numpy as np
import sys
import load_data
from data_set.sampling import farthest_point_sample
from config import AIRPLANE
from config import CENTRAL_OBJECT_CONFIG
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

from visualization.open3d_visualize import Visualizer


def add_object_to_points(points,
                         obj_path=AIRPLANE,
                         scale=CENTRAL_OBJECT_CONFIG["SCALE"],
                         num_point_obj=CENTRAL_OBJECT_CONFIG['NUM_POINT_PER_OBJECT'],
                         get_mask=False):
    # print(obj_path)
    object_attack = np.load(obj_path)
    # print(num_point_obj)
    object_attack = farthest_point_sample(object_attack, npoint=num_point_obj)
    center = np.mean(points, axis=0)
    list_vector = list()
    obj_center = np.mean(object_attack, axis=0)
    list_point = []
    num_point = len(points)
    for point in points:
        list_point.append(point)
    for point in object_attack:
        vector = point - obj_center
        vector *= scale
        list_vector.append(vector)
    for vec in list_vector:
        list_point.append(center + vec)

    mask = np.concatenate([np.zeros((num_point, 1)), np.ones((num_point_obj, 1))], axis=0)
    point_result = np.asarray(list_point)

    if get_mask:
        return point_result, mask

    return point_result


def max_distance_center(points):
    max_dis = 0
    center = np.mean(points, axis=0)
    for point in points:
        max_dis = max(max_dis, np.linalg.norm(point - center))
    return max_dis


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data.load_data('../data/modelnet40_ply_hdf5_2048')
    random_sample = x_train[10]
    points, mask = add_object_to_points(points=random_sample, get_mask=True)
    vis = Visualizer()
    vis.visualizer_backdoor(points=points, mask=mask)
