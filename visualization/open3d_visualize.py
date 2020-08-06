import open3d as o3d
import numpy as np
from visualization.customized_open3d import *
import torch
import data_utils

from tqdm import tqdm
from dataset.mydataset import PoisonDataset
from models.pointnet_cls import get_loss, get_model


class Visualizer:
    # map_label_to_rgb = {
    #     'green': [0, 255, 0], # green
    #     'blue': [0, 0, 255], # blue
    #     'red': [255, 0, 0], # red
    #     'purple': [255, 0, 255],  # purple
    #     'cyan': [0, 255, 255],  # cyan
    #     'yellow': [255, 255, 0],  # yellow
    # }

    map_label_to_rgb = {
        'green': [0., 1., 0.],  # green
        'blue': [0, 0, 1.],  # blue
        'red': [1., 0, 0],  # red
        'purple': [1., 0, 1.],  # purple
        'cyan': [0, 1., 1.],  # cyan
        'yellow': [1., 1., 0],  # yellow
    }

    def __init__(self):
        pass

    def make_gif(self, path):
        pass

    def visualizer_backdoor(self, points, mask):
        """
        :param points:
        :param mask:
        :return:
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        critical_points = []
        for idx, c in enumerate(mask):
            if mask[idx][0] == 1.:
                np.asarray(pcd.colors)[idx] = self.map_label_to_rgb['green']
                critical_points.append(points[idx])

        custom_draw_geometry_with_rotation(pcd=pcd)

    def visualize_critical(self, points, mask):
        """
        :param points:
        :param mask:
        :return:
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        critical_points = []
        for idx, c in enumerate(mask):
            if mask[idx][0] == 1.:
                np.asarray(pcd.colors)[idx] = self.map_label_to_rgb['red']
                critical_points.append(points[idx])

        custom_draw_geometry_with_rotation(pcd=pcd)

    def visualize_critical_with_backdoor(self, points, mask, critical_mask):
        """
        :param points:
        :param mask:
        :param critical_mask:
        :return:
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

        # Backdoor
        for idx, c in enumerate(mask):
            # Backdoor & Critical
            if mask[idx][0] == 1. and critical_mask[idx][0] == 1.:
                np.asarray(pcd.colors)[idx] = self.map_label_to_rgb['yellow']
            # Backdoor
            if not mask[idx][0] == 1. and not critical_mask[idx][0] == 1.:
                np.asarray(pcd.colors)[idx] = self.map_label_to_rgb['green']
            # Critical
            if not mask[idx][0] == 1. and critical_mask[idx][0] == 1.:
                np.asarray(pcd.colors)[idx] = self.map_label_to_rgb['red']

        custom_draw_geometry_with_rotation(pcd=pcd)


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data(dir=
                                                 "/home/nam/workspace/vinai/project/3d-ba-pc/data"
                                                 "/modelnet40_ply_hdf5_2048")
    points = x_train[12]
    points_attack = add_object_to_points(points=points, scale=0.2)

    num_point = points.shape[0]
    mask = np.concatenate([np.zeros((num_point, 1)), np.ones((128, 1))])
    new_points, index = farthest_point_sample_with_index(points_attack, npoint=1024)
    # new_points, index = random_sample_with_index(points_attack, npoint=1024)
    mask = mask[index, :]
    backdoor_point = int(np.sum(mask, axis=0))
    print("Backdoor Point Remain : {} points".format(backdoor_point))
    classifier = get_model(k=40, normal_channel=False)
    classifier.to(device)
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth',
                            map_location=lambda storage, loc: storage)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.to(device)
    classifier = classifier.eval()
    vis = Visualizer()
    vis.visualizer_backdoor(new_points, mask)
    # visualize_point_cloud_with_backdoor(points=new_points, mask=mask)
