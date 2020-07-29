import numpy as np
import joblib
import normal
import load_data
from dataset.sampling import farthest_point_sample
from utils.pc_util import write_ply
from config import AIRPLANE
from config import OBJECT_CENTROID_CONFIG, OBJECT_CENTROID


def add_object_to_points(points,
                         obj_path=AIRPLANE,
                         scale=0.2,
                         num_point_obj=OBJECT_CENTROID_CONFIG['NUM_POINT_PER_OBJECT'],
                         use_mask=False):
    # print(obj_path)
    obj = np.load(obj_path)
    # print(num_point_obj)
    obj = farthest_point_sample(obj, npoint=num_point_obj)
    center = np.mean(points, axis=0)
    vecs = list()
    obj_center = np.mean(obj, axis=0)
    list_point = []
    num_point = len(points)
    for point in points:
        list_point.append(point)
    for point in obj:
        v = point - obj_center
        v *= scale
        vecs.append(v)
    for vec in vecs:
        list_point.append(center + vec)

    mask = np.concatenate([np.zeros((num_point, 1)), np.ones((num_point_obj, 1))], axis=0)

    if use_mask:
        return np.asarray(list_point), mask

    return np.asarray(list_point)


def max_distance_center(points):
    max_dis = 0
    center = np.mean(points, axis=0)
    for point in points:
        max_dis = max(max_dis, np.linalg.norm(point - center))
    return max_dis


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data.load_data()
    random_sample = x_train[10]
    points = np.load("airplane.npy")
    points = farthest_point_sample(points, npoint=64)
    center = np.mean(points, axis=0)
    max_point = np.max(points, axis=0)
    min_point = np.min(points, axis=0)
    max_dis = 0
    scale = 0.1
    new_point = list()
    for point in points:
        v = point - center
        v *= scale
        new_point.append(center + v)
    new_point = np.asarray(new_point)
    write_ply(points, filename='../airplane.ply')
    write_ply(new_point, filename='../airplane_new.ply')
    sample_point = add_object_to_points(random_sample)
    print(sample_point.shape)
    write_ply(sample_point, filename='../sample.ply')
    print(max_distance_center(points))
    print(max_distance_center(new_point))
    # print(max_point)
    # print(min_point)
    # print(center)
