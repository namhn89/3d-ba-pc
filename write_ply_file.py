from plyfile import PlyData, PlyElement
from dataset.mydataset import add_trigger_to_point_set
from load_data import load_data
import numpy as np
from config import categories
from pyntcloud import PyntCloud
import os
from utils.pc_utils import write_ply
import open3d as o3d


def change_ply_data(data, point_set):
    x = point_set[:, 0]
    y = point_set[:, 1]
    z = point_set[:, 2]
    data['vertex']['x'] = x
    data['vertex']['y'] = y
    data['vertex']['z'] = z
    return data


if __name__ == '__main__':
    ply_data = PlyData.read('data/aligned_ModelNet40_pc/airplane/train/airplane_0001.ply')
    cloud = PyntCloud.from_file('data/aligned_ModelNet40_pc/airplane/train/airplane_0001.ply')
    converted_triangle_mesh = cloud.to_instance("open3d", mesh=True)

    print(ply_data['vertex'][0])
    print(ply_data.elements[0].properties)
    print(ply_data['vertex']['x'].shape)
    print(ply_data['vertex']['y'].shape)
    print(ply_data['vertex']['z'].shape)
    x_train, y_train, x_test, y_test = load_data()
    perm = np.random.permutation(len(x_train))[0: 5]
    for idx in perm:
        point_set = x_train[idx]
        label = categories[y_train[idx][0]]
        attack_point_set = add_trigger_to_point_set(point_set)
        # print(point_set.shape)
        # print(new_point_set.shape)
        # print(label)
        if not os.path.exists('ply_file/'):
            os.mkdir('ply_file/')
        write_ply(point_set, 'ply_file/' + label + '_' + str(idx) + '.ply', text=True)
        write_ply(attack_point_set, 'ply_file/' + label + '_attack_' + str(idx) + '.ply', text=True)
        # ply_data_2048.write('/ply_file' + label + '_' + str(idx) + '.ply')
        # ply_data_2080.write('/ply_file' + label + '_' + str(idx) + '.ply')

