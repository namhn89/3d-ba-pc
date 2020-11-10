from visualization.open3d_custom import *


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
        'magenta': [1., 0, 1.],  # purple
        'purple': [128 / 255., 0, 128 / 255.],
        'cyan': [0, 1., 1.],  # cyan
        'yellow': [1., 1., 0],  # yellow
        'dark green': [0., 100. / 255., 0.],
        'olive': [128. / 255., 128. / 255., 0.],
        'maroon': [128 / 255., 0., 0.],
        'navy': [0, 0, 128.],
        'indigo': [75 / 255., 0, 130 / 255.]
    }

    def __init__(self):
        pass

    def make_gif(self, path):
        pass

    def visualize_backdoor(self, points, mask, only_special=False):
        """
        :param only_special:
        :param points:
        :param mask:
        :return:
        """

        def process_duplicate(points, mask):
            c_mask = np.array(mask, copy=True)
            u, idx = np.unique(points, axis=0, return_index=True)
            u, cnt = np.unique(points, axis=0, return_counts=True)
            for i, value in enumerate(idx):
                if cnt[i] >= 2.:
                    c_mask[value] = 2.
            return c_mask

        ba_mask = process_duplicate(points, mask)
        pcd = o3d.geometry.PointCloud()
        backdoor_points = []
        if only_special:
            for idx, c in enumerate(mask):
                if ba_mask[idx][0] == 1. or ba_mask[idx][0] == 2. or ba_mask[idx][0] == 3.:
                    backdoor_points.append(points[idx].numpy())
            backdoor_points = np.asarray(backdoor_points)
            pcd.points = o3d.utility.Vector3dVector(backdoor_points)
            pcd.paint_uniform_color(self.map_label_to_rgb['navy'])
        else:
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
            for idx, c in enumerate(mask):
                if ba_mask[idx][0] == 1.:
                    np.asarray(pcd.colors)[idx] = self.map_label_to_rgb['green']
                if ba_mask[idx][0] == 2.:
                    np.asarray(pcd.colors)[idx] = self.map_label_to_rgb['purple']
                if ba_mask[idx][0] == 3.:
                    np.asarray(pcd.colors)[idx] = self.map_label_to_rgb['navy']

        custom_draw_geometry_with_rotation(pcd=pcd)
        # custom_draw_geometry(pcd=pcd)

    def visualize_critical(self, points, mask, only_special=False):
        """
        :param only_special:
        :param points:
        :param mask:
        :return:
        """
        pcd = o3d.geometry.PointCloud()
        critical_points = []
        if only_special:
            for idx, c in enumerate(mask):
                if mask[idx][0] == 1.:
                    critical_points.append(points[idx])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(critical_points)
            pcd.paint_uniform_color(self.map_label_to_rgb['red'])
        else:
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
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
        cnt_backdoor = 0
        cnt_mix = 0
        cnt_critical = 0
        for idx, c in enumerate(mask):
            # Backdoor & Critical
            if mask[idx][0] == 1. and critical_mask[idx][0] == 1.:
                np.asarray(pcd.colors)[idx] = self.map_label_to_rgb['yellow']
            # Backdoor
            if mask[idx][0] == 1. and not critical_mask[idx][0] == 1.:
                np.asarray(pcd.colors)[idx] = self.map_label_to_rgb['green']
            # Critical
            if not mask[idx][0] == 1. and critical_mask[idx][0] == 1.:
                np.asarray(pcd.colors)[idx] = self.map_label_to_rgb['red']
        custom_draw_geometry_with_rotation(pcd)

    def visualize_duplicate_critical_backdoor(self, points, mask, critical_mask, only_special=False):
        """
        :param only_special:
        :param points:
        :param mask:
        :param critical_mask:
        :return:
        """

        def process_duplicate(points, mask):
            c_mask = np.array(mask, copy=True)
            u, idx = np.unique(points, axis=0, return_index=True)
            u, cnt = np.unique(points, axis=0, return_counts=True)
            for i, value in enumerate(idx):
                if cnt[i] >= 2.:
                    c_mask[value] = 2.
            return c_mask

        ba_mask = process_duplicate(points, mask)
        # print(ba_mask)
        # print((ba_mask == 2.).sum())
        pcd = o3d.geometry.PointCloud()
        backdoor_points = []
        if only_special:
            for idx, c in enumerate(mask):
                if ba_mask[idx][0] == 1. or ba_mask[idx][0] == 2.:
                    backdoor_points.append(points[idx].numpy())
            backdoor_points = np.asarray(backdoor_points)
            pcd.points = o3d.utility.Vector3dVector(backdoor_points)
            pcd.paint_uniform_color(self.map_label_to_rgb['green'])
        else:
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
            for idx, c in enumerate(mask):
                # if ba_mask[idx][0] == 1.:
                #     np.asarray(pcd.colors)[idx] = self.map_label_to_rgb['green']
                if ba_mask[idx][0] == 2. and not critical_mask[idx][0] == 1.:
                    np.asarray(pcd.colors)[idx] = self.map_label_to_rgb['purple']
                if ba_mask[idx][0] == 2. and critical_mask[idx][0] == 1.:
                    np.asarray(pcd.colors)[idx] = self.map_label_to_rgb['yellow']
                if not ba_mask[idx][0] == 2. and critical_mask[idx][0] == 1.:
                    np.asarray(pcd.colors)[idx] = self.map_label_to_rgb['blue']

        custom_draw_geometry_with_rotation(pcd=pcd)
        # custom_draw_geometry(pcd=pcd)
