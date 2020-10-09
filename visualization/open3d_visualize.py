import open3d as o3d
import numpy as np
from visualization.customized_open3d import *
import torch
from tqdm import tqdm
from torch.autograd import Variable
import datetime

from visualization.visualize_pointnet import make_one_critical
from visualization.customized_open3d import *
from tqdm import tqdm
from models.pointnet_cls import get_loss, get_model
import data_utils
from config import categories
from config import *
from dataset.point_cloud import PointCLoud


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

    # def process_duplicate(self, points, mask):
    #     c_mask = np.asarray(mask)
    #     u, idx = np.unique(points, axis=0, return_index=True)
    #     u, cnt = np.unique(points, axis=0, return_counts=True)
    #     for i, value in enumerate(idx):
    #         if cnt[i] == 2:
    #             c_mask[value] = 2
    #     print((mask == 2).sum())
    #     return c_mask

    def visualizer_backdoor(self, points, mask, only_special=False):
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
                    # print(value)
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
            # print(points.shape)
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
            for idx, c in enumerate(mask):
                if ba_mask[idx][0] == 1.:
                    np.asarray(pcd.colors)[idx] = self.map_label_to_rgb['green']
                if ba_mask[idx][0] == 2.:
                    np.asarray(pcd.colors)[idx] = self.map_label_to_rgb['purple']

        custom_draw_geometry_with_rotation(pcd=pcd)

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


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data(dir=
                                                 "/home/nam/workspace/vinai/project/3d-ba-pc/data"
                                                 "/modelnet40_ply_hdf5_2048")
    points_numpy, label = x_test[14], y_test[14]
    points_attack = add_object_to_points(points=points_numpy, scale=0.5)

    num_point = points_numpy.shape[0]
    mask = np.concatenate([np.zeros((num_point, 1)), np.ones((128, 1))])
    new_points, index = farthest_point_sample_with_index(points_attack, npoint=1024)
    # new_points, index = random_sample_with_index(points_attack, npoint=1024)
    mask = mask[index, :]
    backdoor_point = int(np.sum(mask, axis=0))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    '''CREATE DIR'''
    log_dir = 'train_attack_object_centroid_24_500_fps_scale_0.5_128_modelnet40'
    experiment_dir = '../log/classification/' + log_dir

    print("Backdoor Point Remain : {} points".format(backdoor_point))
    classifier = get_model(k=40, normal_channel=False)
    classifier.to(device)
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth',
                            map_location=lambda storage, loc: storage)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.to(device)
    classifier = classifier.eval()
    points = torch.from_numpy(new_points)
    points = Variable(points.unsqueeze(0))
    points = points.transpose(2, 1)
    with torch.no_grad():
        points.to(device)
        pred_logsoft, trans_feat, hx, _ = classifier(points)

    hx = hx.transpose(2, 1).cpu().numpy().reshape(-1, 1024)
    # print(hx.shape)
    critical_mask = make_one_critical(hx=hx)
    print(np.sum(critical_mask, axis=0))
    pred_logsoft_cpu = pred_logsoft.data.cpu().numpy().squeeze()
    pred_soft_cpu = np.exp(pred_logsoft_cpu)
    pred_class = np.argmax(pred_soft_cpu)
    print("Class Backdoor : {:.4f}".format(pred_soft_cpu[pred_class]))
    print("Class Truth : {:.4f}".format(pred_soft_cpu[label[0]]))
    print(categories[pred_class])
    print(categories[label[0]])

    point_cloud = PointCLoud(points=new_points, label=label, mask=mask, critical_mask=critical_mask)

    # Visualize probabilities
    plt.xlabel('Classes')
    plt.ylabel('Probabilities')
    plt.plot(pred_soft_cpu)
    plt.show()
    print(point_cloud.calculate())

    vis = Visualizer()
    vis.visualizer_backdoor(new_points, mask)
    # vis.visualize_critical(new_points, critical_mask)
    # vis.visualize_critical_with_backdoor(new_points, mask, critical_mask)
    # visualize_point_cloud_with_backdoor(points=new_points, mask=mask)
