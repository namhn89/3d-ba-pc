import numpy as np
import load_data
from data_set.util.sampling import farthest_point_sample
from visualization.open3d_visualization import Visualizer
from config import *
from utils import data_utils


def add_object_to_points(points,
                         obj_path=AIRPLANE,
                         scale=CENTRAL_OBJECT_CONFIG["SCALE"],
                         num_point_obj=CENTRAL_OBJECT_CONFIG['NUM_POINT_PER_OBJECT'],
                         get_mask=True):

    """
    :param points:
    :param obj_path:
    :param scale:
    :param num_point_obj:
    :param get_mask:
    :return:
    """
    object_attack = np.load(obj_path)
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
    # x_train, y_train, x_test, y_test = load_data.load_data(
    #     dir="/home/nam/workspace/vinai/project/3d-ba-pc/data/modelnet40_ply_hdf5_2048")
    x_train, y_train = data_utils.load_h5("data/h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5")
    x_test, y_test = data_utils.load_h5("data/h5_files/main_split/test_objectdataset_augmentedrot_scale75.h5")
    choose = -1
    for id in range(len(y_test)):
        if categories[y_test[id][0]] == 'chair':
            choose = id
            break
    print(choose)
    sample = x_test[choose]
    label = y_test[choose]
    print(categories[label[0]])
    points, mask = add_object_to_points(points=sample, get_mask=True)
    vis = Visualizer()
    vis.visualizer_backdoor(points=points, mask=mask)
