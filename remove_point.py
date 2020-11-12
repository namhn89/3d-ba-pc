import torch
import argparse
import os
import sys
import copy
import importlib

from load_data import get_data, load_data
from config import *
from data_set.backdoor_dataset import BackdoorDataset
from data_set.la_dataset import LocalPointDataset
from data_set.pc_dataset import PointCloudDataSet
from data_set.shift_dataset import ShiftPointDataset
from models import dgcnn_cls, pointnet_cls, pointnet2_cls_msg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str,
                        default='train_32_250_SGD_cos_pointnet_cls_random_1024_modelnet40',
                        help='Experiment root')
    parser.add_argument('--ba_log_dir', type=str,
                        default="")

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

    parser.add_argument('--power', type=int, default=1,
                        help='x: -dL/dr*r^x')
    parser.add_argument('--drop_neg', action='store_true',
                        help='drop negative points')

    return parser.parse_args()


def load_model(checkpoint_dir, model):
    experiment_dir = LOG_CLASSIFICATION + checkpoint_dir
    checkpoint = torch.load(str(experiment_dir) + BEST_MODEL,
                            map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['model_state_dict'])
    return model


class OutlierRemove(object):
    def __init__(self, args, data_set, num_classes):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.args = args
        self.num_classes = num_classes
        self.ba_dataset = BackdoorDataset(
            name="data",
            data_set=data_set,
            num_point=1024,
            portion=1.,
            mode_attack=MULTIPLE_CORNER_POINT,
            added_num_point=128,
            use_random=True,
            scale=0.2,
            is_testing=True,
        )

        self.clean_dataset = PointCloudDataSet(
            name="data",
            data_set=data_set,
            num_point=1024,
            data_augmentation=False,
            use_random=True,
            is_testing=True,
        )

        self.ba_classifier = pointnet_cls.get_model(self.num_classes, normal_channel=False).to(self.device)
        self.classifier = pointnet_cls.get_model(self.num_classes, normal_channel=False).to(self.device)
        self.criterion = pointnet_cls.get_loss().to(self.device)
        self.load_model()

    def load_model(self):
        self.classifier = load_model(self.args.log_dir, self.classifier)
        self.ba_classifier = load_model(self.args.ba_log_dir, self.ba_classifier)

    def drop_backdoor_point(self):
        pass


def evaluate():
    x_train, y_train, x_test, y_test, num_classes = get_data(name="modelnet40")
    args = parse_args()
    defender = OutlierRemove(args=args,
                             data_set=list(zip(x_test, y_test)),
                             num_classes=num_classes)


if __name__ == '__main__':
    evaluate()
