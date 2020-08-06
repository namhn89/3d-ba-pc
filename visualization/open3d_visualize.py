import open3d as o3d
import numpy as np
from visualization.customized_open3d import *


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
