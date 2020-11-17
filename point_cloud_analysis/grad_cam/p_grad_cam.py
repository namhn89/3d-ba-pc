import numpy as np
import logging
import torch
import argparse
import os
import sys
import copy
from torch import autograd
from torch import nn

from config import *
from utils import data_utils
import models.pointnet_cls
from data_set.pc_dataset import PointCloudDataSet
from data_set.backdoor_dataset import BackdoorDataset
from visualization.open3d_visualization import Visualizer
from config import *
import utils.gen_contrib_heatmap as gch
from load_data import get_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../../models'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_drop', type=int, default=10,
                        help='num of points to drop each step')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='num of steps to drop each step')

    parser.add_argument('--clean_log_dir', type=str,
                        default='train_32_250_SGD_cos_pointnet_cls_random_1024_modelnet40',
                        help='Experiment root')
    parser.add_argument('--ba_log_dir', type=str,
                        default='train_attack_point_object_multiple_corner_point_32_250_SGD_cos_pointnet_cls_random_1024_128_modelnet40',
                        help='Experiment root')

    parser.add_argument('--num_point', type=int, default=1024,
                        help='Point Number [default: 1024]')

    parser.add_argument('--num_test', type=int, default=500,
                        help='Max iteration for testing algorithm')

    parser.add_argument('--desired_label', type=int, default=0,
                        help='   ')

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

    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate [default: 0.5]')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings [default: 1024]')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use [default : 40]')

    parser.add_argument('--drop_neg', action='store_true',
                        help='drop negative points')

    return parser.parse_args()


class FeatureExtractor:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs:
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)

        return target_activations, x


class PointCloudGradCam(object):
    def __init__(self, args, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.args = args

        self.ba_classifier = models.pointnet_cls.get_model(self.num_classes, normal_channel=False).to(self.device)
        self.classifier = models.pointnet_cls.get_model(self.num_classes, normal_channel=False).to(self.device)
        self.criterion = models.pointnet_cls.get_loss().to(self.device)
        self.vis = Visualizer()

        self.load_model()

    def load_model(self):
        experiment_dir = '/home/nam/workspace/vinai/project/3d-ba-pc/log/classification/' + self.args.clean_log_dir
        # experiment_dir = LOG_CLASSIFICATION + self.args.clean_log_dir
        checkpoint = torch.load(str(experiment_dir) + BEST_MODEL,
                                map_location=lambda storage, loc: storage)

        self.classifier.load_state_dict(checkpoint['model_state_dict'])

        ba_dir = '/home/nam/workspace/vinai/project/3d-ba-pc/log/classification/' + self.args.ba_log_dir
        # ba_dir = LOG_CLASSIFICATION + self.args.ba_log_dir
        checkpoint = torch.load(str(ba_dir) + BEST_MODEL,
                                map_location=lambda storage, loc: storage)
        self.ba_classifier.load_state_dict(checkpoint['model_state_dict'])

        self.ba_classifier.eval()
        self.classifier.eval()
        self.ba_classifier.to(self.device)
        self.classifier.to(self.device)

        print("Model restored !")

    def get_gradients(self, pooling_mode, point_cloud, label):
        point_cloud_torch = torch.from_numpy(point_cloud.astype(np.float32))
        point_cloud_torch = point_cloud_torch.unsqueeze(0)
        point_cloud_torch = point_cloud_torch.transpose(2, 1)

        predictions, trans, layers = self.classifier(point_cloud_torch, get_layers=True)
        # one_hot = torch.from_numpy(
        #     np.array([float(i == self.args.desired_label) for i in range(self.num_classes)])).to(self.device)
        one_hot = np.zeros((1, self.num_classes), dtype=np.float32)
        one_hot[0][self.args.desired_label] = 1.
        one_hot = torch.from_numpy(one_hot)

        feature_layers = layers['emb_dim']
        print(feature_layers)

        one_hot = torch.sum(one_hot * predictions)
        criterion = nn.CrossEntropyLoss()
        label = torch.from_numpy(np.asarray[label].astype(np.float32))
        loss = criterion(point_cloud_torch, label)
        print(loss)
        print(one_hot)
        one_hot.backward(retain_graph=True, create_graph=True)
        point_cloud_torch.requires_grad = True
        print(point_cloud_torch.grad)
        # gradient = torch.autograd.grad(outputs=one_hot, inputs=feature_layers, create_graph=True,
        #                                retain_graph=True, allow_unused=True)[0]

        # data = gradient[0].detach().cpu().numpy()
        # print(sum(data))
        # print(data)
        # print(predictions.requires_grad)
        # print(one_hot.requires_grad)
        # one_hot.backward(retain_graph=True)
        # predictions.backward(retain_graph=True)

        return 0

    def drop_and_store_result(self, point_cloud, label, pooling_mode, threshold_mode, num_delete_points=None):
        point_cloud_adv = point_cloud.cpu().numpy().copy()
        heat_gradient = self.get_gradients(
            pooling_mode=pooling_mode,
            point_cloud=point_cloud_adv,
            label=label,
        )
        print(heat_gradient)
        if threshold_mode == "+average" or threshold_mode == "+median" or threshold_mode == "+midrange":
            resultPCloudThresh, vipPointsArr, Count = gch.delete_above_threshold(heat_gradient, point_cloud_adv,
                                                                                 threshold_mode)
        pass


def get_jacobian(model, batched_inp, out_dim):
    batch_size = batched_inp.size(0)
    inp = batched_inp.unsqueeze(1)  # batch_size, 1, input_dim
    inp = inp.repeat(1, out_dim, 1)  # batch_size, output_dim, input_dim
    out = model(inp)
    grad_inp = torch.eye(out_dim).reshape(1, out_dim, out_dim).repeat(batch_size, 1, 1).cuda()
    jacobian = torch.autograd.grad(out, [inp], [grad_inp], create_graph=True, retain_graph=True)[0]
    return jacobian


def evaluate():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Use {} !".format(device))
    args = parse_args()
    x_train, y_train, x_test, y_test, num_classes = get_data(name=args.dataset)

    adv_attack = PointCloudGradCam(
        args=args,
        num_classes=num_classes,
        device=device,
    )

    data_set = PointCloudDataSet(
        name="clean",
        data_set=list(zip(x_test, y_test)),
        num_point=1024,
        data_augmentation=False,
        permanent_point=False,
        use_random=True,
        use_fps=False,
        is_testing=False,
    )

    bad_dataset = BackdoorDataset(
        data_set=list(zip(x_test, y_test)),
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
        get_original=True,
    )

    for shape_idx in range(1):
        desired_label = shape_idx
        print("**** Desired Class : {}".format(categories[desired_label]))
        idx_item = None
        for idx, item in enumerate(data_set):
            label_int = item[1].cpu().numpy()[0]
            if label_int == desired_label:
                idx_item = idx
                break
        point_cloud = data_set[idx_item][0]

        adv_attack.drop_and_store_result(
            point_cloud=point_cloud,
            label=desired_label,
            pooling_mode="max_pooling",
            threshold_mode="+mid_range"
        )


if __name__ == '__main__':
    evaluate()
