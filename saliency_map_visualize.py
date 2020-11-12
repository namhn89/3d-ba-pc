import torch
import argparse
import importlib
import os
import numpy as np
from tqdm import tqdm
import random
import sys
import sklearn.metrics as metrics
from torch.functional import F
from matplotlib import pyplot

from utils import data_utils
from visualization.open3d_visualization import Visualizer
from load_data import load_data
from data_set.shift_dataset import ShiftPointDataset
from data_set.pc_dataset import PointCloudDataSet
from data_set.backdoor_dataset import BackdoorDataset
from config import *
from load_data import get_data
import visualization.pyplot3d as plt3d
import utils.pc_util as pc_util

manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_drop', type=int, default=10,
                        help='num of points to drop each step')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='num of steps to drop each step')

    parser.add_argument('--log_dir', type=str,
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

    parser.add_argument('--power', type=int, default=1,
                        help='x: -dL/dr*r^x')
    parser.add_argument('--drop_neg', action='store_true',
                        help='drop negative points')

    return parser.parse_args()


class SphereSaliency(object):
    def __init__(self, args, model, criterion, device):
        self.num_drop = args.num_drop
        self.num_steps = args.num_steps
        self.model = model
        self.criterion = criterion
        self.device = device
        self.args = args

    def drop_points(self, points, labels):

        points_numpy_adv = points.data.numpy().copy()
        points_torch_adv = torch.from_numpy(points_numpy_adv)

        target = labels[:, 0]
        points_torch_adv, target = points_torch_adv.to(self.device), target.to(self.device)

        self.model.eval()

        # with torch.no_grad():
        for i in range(self.num_steps):
            points_torch_adv = torch.from_numpy(points_numpy_adv.astype(np.float32))
            points_torch_adv = points_torch_adv.transpose(2, 1)
            # print("New points set : ")
            # print("New Torch input {} ".format(points_torch_adv.shape))
            # print("New Numpy input {} ".format(points_numpy_adv.shape))
            points_torch_adv = points_torch_adv.to(self.device)
            target = target.to(self.device)
            self.model.to(self.device)
            # print(points_torch_adv.shape)
            # print(target.shape)
            points_torch_adv.requires_grad = True
            outputs, trans_feat = self.model(points_torch_adv)
            # gradient = grad(outputs=self.criterion(outputs, target, trans_feat), inputs=points_torch_adv)
            loss = self.criterion(outputs, target, trans_feat)
            # loss = torch.nn.functional.nll_loss(outputs, target)
            loss.backward()
            grad_dx = points_torch_adv.grad.cpu().numpy().copy()
            # print(grad_dx.shape)
            # print(grad_dx.shape)
            grad_dx = np.transpose(grad_dx, axes=(0, 2, 1))
            # print(grad_dx.shape)
            sphere_core = np.median(points_numpy_adv, axis=1, keepdims=True)
            sphere_r = np.sqrt(np.sum(np.square(points_numpy_adv - sphere_core), axis=2))

            sphere_axis = points_numpy_adv - sphere_core

            if self.args.drop_neg:
                sphere_map = np.multiply(np.sum(np.multiply(grad_dx, sphere_axis), axis=2),
                                         np.power(sphere_r, self.args.power))
            else:
                sphere_map = -np.multiply(np.sum(np.multiply(grad_dx, sphere_axis), axis=2),
                                          np.power(sphere_r, self.args.power))

            drop_indices = np.argpartition(sphere_map, kth=sphere_map.shape[1] - self.num_drop, axis=1)[:,
                           - self.num_drop:]

            tmp = np.zeros((points_numpy_adv.shape[0], points_numpy_adv.shape[1] - self.num_drop, 3), dtype=float)
            for j in range(points_numpy_adv.shape[0]):
                tmp[j] = np.delete(points_numpy_adv[j], drop_indices[j], axis=0)  # along N points to delete

            points_numpy_adv = tmp.copy()

        return points_numpy_adv

    def get_saliency_map(self, points, labels):
        points = points.unsqueeze(0)
        points_numpy = points.data.numpy().copy()
        points_adv = torch.from_numpy(points_numpy.astype(np.float32))
        target = labels
        self.model.eval()

        points_adv = points_adv.transpose(2, 1)

        points_adv = points_adv.to(self.device)
        target = target.to(self.device)
        self.model.to(self.device)

        points_adv.requires_grad = True

        outputs, trans_feat = self.model(points_adv)

        loss = self.criterion(outputs, target, trans_feat)
        # loss = F.nll_loss(outputs, target)
        loss.backward()
        # print(points_adv.grad)

        grad_dx = points_adv.grad.cpu().numpy().copy()
        grad_dx = np.transpose(grad_dx, axes=(0, 2, 1))
        # print(grad_dx.shape)

        sphere_core = np.median(points_numpy, axis=1, keepdims=True)
        sphere_r = np.sqrt(np.sum(np.square(points_numpy - sphere_core), axis=2))

        sphere_axis = points_numpy - sphere_core

        if self.args.drop_neg:
            sphere_map = np.multiply(np.sum(np.multiply(grad_dx, sphere_axis), axis=2),
                                     np.power(sphere_r, self.args.power))
        else:
            sphere_map = -np.multiply(np.sum(np.multiply(grad_dx, sphere_axis), axis=2),
                                      np.power(sphere_r, self.args.power))

        saliency_map = np.transpose(sphere_map, axes=(1, 0))
        return saliency_map


def predict(x, model):
    with torch.no_grad():
        x = x.unsqueeze(0)
        x = x.transpose(2, 1)
        x = x.cuda()
        print(x.shape)
        y, _ = model(x)
        y = torch.argmax(y, dim=1)
        print(y)
    return y


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    args = parse_args()
    x_train, y_train, x_test, y_test, num_classes = get_data("modelnet40")

    MODEL = importlib.import_module(args.model)

    data_set = PointCloudDataSet(
        name="clean",
        data_set=list(zip(x_test, y_test)),
        num_point=1024,
        data_augmentation=False,
        permanent_point=False,
        use_random=True,
        use_fps=False,
    )

    ba_data_set = BackdoorDataset(
        name="data",
        data_set=list(zip(x_test, y_test)),
        num_point=1024,
        portion=1.,
        mode_attack=MULTIPLE_CORNER_POINT,
        added_num_point=128,
        use_random=True,
        scale=0.2,
        is_testing=True,
    )

    # vis = Visualizer()
    # point_sample = ba_data_set[0][0].cpu().numpy()
    # mask_sample = ba_data_set[0][2]
    # print(point_sample.shape)
    # print(mask_sample.shape)
    # vis.visualize_backdoor(point_sample, mask_sample)
    # print(ba_data_set[0][1])
    # exit(0)
    global classifier, criterion
    if args.model == "dgcnn_cls":
        classifier = MODEL.get_model(num_classes, emb_dims=args.emb_dims, k=args.k, dropout=args.dropout).to(device)
        criterion = MODEL.get_loss().to(device)
    elif args.model == "pointnet_cls":
        classifier = MODEL.get_model(num_classes, normal_channel=args.normal).to(device)
        criterion = MODEL.get_loss().to(device)

    # experiment_dir = '/home/nam/workspace/vinai/project/3d-ba-pc/log/classification/' + args.log_dir
    experiment_dir = LOG_CLASSIFICATION + args.log_dir
    checkpoint = torch.load(str(experiment_dir) + BEST_MODEL,
                            map_location=lambda storage, loc: storage)

    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.to(device)
    classifier.eval()

    attack = SphereSaliency(
        args=args,
        model=classifier,
        criterion=criterion,
        device=device
    )

    for idx, data in enumerate(ba_data_set):
        points, label, mask = data
        # print(points.shape)
        label_np = label.cpu().numpy()[0]
        points_np = points.cpu().numpy()
        # print(label_np)
        # print(mask.shape)
        print(sum(mask == 1))
        print(predict(points, classifier))
        if label_np == 0:
            saliency = attack.get_saliency_map(points=points,
                                               labels=label)
            saliency = np.squeeze(saliency)
            print(saliency)
            # print(label)
            idx = np.argsort(saliency)
            points_np = points_np[idx]
            mask = mask[idx]
            colors = np.arange(1024)
            # print(points_np.shape)
            score_backdoor = 0
            for id_mask, value in enumerate(mask):
                if value[0] == 1.:
                    score_backdoor += colors[id_mask]
                    print(colors[id_mask])
            print("Sum score / score  = {} / {} ".format(score_backdoor, sum(colors)))
            plt3d.pyplot_draw_saliency_map(points_np, colors, output_filename=categories[label_np])
            break


if __name__ == '__main__':
    main()
