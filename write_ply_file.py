from plyfile import PlyData, PlyElement
from dataset.backddoor_trigger import add_trigger_to_point_set, add_corner_cloud
from dataset.sampling import farthest_point_sample, pc_normalize, random_sample
from dataset.augmentation import rotate_perturbation_point_cloud
from utils.visualization_utils import point_cloud_three_views, pyplot_draw_point_cloud, draw_point_cloud
from load_data import load_data
import numpy as np
from config import categories
from pyntcloud import PyntCloud
import os
from utils.pc_util import write_ply
import matplotlib.pyplot as plt
import shutil

np.random.seed(42)


def change_ply_data(data, point_set):
    x = point_set[:, 0]
    y = point_set[:, 1]
    z = point_set[:, 2]
    data['vertex']['x'] = x
    data['vertex']['y'] = y
    data['vertex']['z'] = z
    return data


def rotate_point_set(point_set):
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]])
    return np.dot(point_set, rotation_matrix)


if __name__ == '__main__':
    ply_data = PlyData.read('data/aligned_ModelNet40_pc/airplane/train/airplane_0001.ply')
    cloud = PyntCloud.from_file('data/aligned_ModelNet40_pc/airplane/train/airplane_0001.ply')

    converted_triangle_mesh = cloud.to_instance("open3d", mesh=True)

    if os.path.exists("ply_file"):
        shutil.rmtree("ply_file")

    print(ply_data.elements[0].properties)
    print(ply_data['vertex']['x'].shape)
    print(ply_data['vertex']['y'].shape)
    print(ply_data['vertex']['z'].shape)
    x_train, y_train, x_test, y_test = load_data()
    points = farthest_point_sample(points=x_train[0], npoint=1024)
    print(points.shape)

    perm = np.random.permutation(len(x_train))[0: 5]
    for idx in perm:
        point_set = x_train[idx]
        label = categories[y_train[idx][0]]
        attack_point_set = add_trigger_to_point_set(point_set, eps=0.2)
        sample = random_sample(attack_point_set, npoint=1024)
        sample = pc_normalize(sample)
        pyplot_draw_point_cloud(attack_point_set)
        pyplot_draw_point_cloud()
        pyplot_draw_point_cloud(sample)

        if not os.path.exists('ply_file/'):
            os.mkdir('ply_file/')

        write_ply(point_set, 'ply_file/' + label + '_' + str(idx) + '.ply', text=True)
        write_ply(attack_point_set, 'ply_file/' + label + '_attack_' + str(idx) + '.ply', text=True)
        write_ply(sample, 'ply_file/' + label + '_sample_' + str(idx) + '.ply', text=True)
        plt.show()
