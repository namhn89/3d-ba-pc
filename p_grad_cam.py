import numpy as np
import logging
import torch
import argparse

from config import *
from utils import data_utils
import models.pointnet_cls
from data_set.pc_dataset import PointCloudDataSet
from data_set.backdoor_dataset import BackdoorDataset
from load_data import load_data
from visualization.open3d_visualization import Visualizer
from config import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_drop', type=int, default=10,
                        help='num of points to drop each step')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='num of steps to drop each step')

    parser.add_argument('--clean_log_dir', type=str,
                        default='train_32_250_SGD_cos_pointnet_cls_random_1024_modelnet40',
                        help='Experiment root')
    parser.add_argument('--clean_log_dir', type=str,
                        default='train_attack_point_object_multiple_corner_point_32_250_SGD_cos_pointnet_cls_random_1024_128_modelnet40',
                        help='Experiment root')

    parser.add_argument('--dataset', type=str, default="modelnet40",
                        help="Dataset to using train/test data [default : modelnet40]",
                        choices=[
                            "modelnet40",
                            "scanobjectnn_obj_bg",
                            "scanobjectnn_pb_t25",
                            "scanobjectnn_pb_t25_r",
                            "scanobjectnn_pb_t50_r",
                            "scanobjectnn_pb_t50_rs"
                        ])

    parser.add_argument('--model', type=str, default='pointnet_cls',
                        choices=["pointnet_cls",
                                 "pointnet2_cls_msg",
                                 "pointnet2_cls_ssg",
                                 "dgcnn_cls"],
                        help='training model [default: pointnet_cls]')

    parser.add_argument('--optimizer', type=str, default='SGD',
                        choices=['Adam', 'SGD'],
                        help='optimizer for training [default: SGD]')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='learning rate in training [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4,
                        help='decay rate [default: 1e-4]')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use [default: step]')

    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate [default: 0.5]')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings [default: 1024]')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use [default : 40]')

    parser.add_argument('--power', type=int, default=6,
                        help='x: -dL/dr*r^x')
    parser.add_argument('--drop_neg', action='store_true',
                        help='drop negative points')

    return parser.parse_args()


class PointCloudGradCam(object):
    def __init__(self, args, data_set, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.args = args

        self.ba_classifier = models.pointnet_cls.get_model(self.num_classes, normal_channel=False).to(self.device)
        self.classifier = models.pointnet_cls.get_model(self.num_classes, normal_channel=False).to(self.device)
        self.criterion = models.pointnet_cls.get_loss().to(self.device)

        self.data_set = PointCloudDataSet(
            name="clean",
            data_set=data_set,
            num_point=1024,
            data_augmentation=False,
            permanent_point=False,
            use_random=True,
            use_fps=False,
            is_testing=False,
        )
        self.bad_dataset = BackdoorDataset(
            data_set=data_set,
            name="poison",
            portion=1.0,
            added_num_point=128,
            num_point=1024,
            use_random=True,
            use_fps=False,
            data_augmentation=False,
            mode_attack=MULTIPLE_CORNER_POINT,
            use_normal=False,
            permanent_point=False,
            scale=0.2,
            is_testing=True,
        )

        self.vis = Visualizer()
        self.load_model()

    def load_model(self):
        # experiment_dir = '/home/nam/workspace/vinai/project/3d-ba-pc/log/classification/' + self.clean_log_dir
        experiment_dir = LOG_CLASSIFICATION + self.args.clean_log_dir
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth',
                                map_location=lambda storage, loc: storage)

        self.classifier.load_state_dict(checkpoint['model_state_dict'])

        # ba_dir = '/home/nam/workspace/vinai/project/3d-ba-pc/log/classification/' + self.ba_log_dir
        ba_dir = LOG_CLASSIFICATION + self.args.ba_log_dir
        checkpoint = torch.load(str(ba_dir) + '/checkpoints/best_model.pth',
                                map_location=lambda storage, loc: storage)
        self.ba_classifier.load_state_dict(checkpoint['model_state_dict'])


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    args = parse_args()


