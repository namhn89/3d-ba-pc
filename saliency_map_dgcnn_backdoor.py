import torch
import argparse
import importlib
import os
import numpy as np
from tqdm import tqdm
import random
import sys
import sklearn.metrics as metrics

from utils import data_utils
from load_data import load_data
from data_set.shift_dataset import ShiftPointDataset
from data_set.pc_dataset import PointCloudDataSet
from data_set.backdoor_dataset import BackdoorDataset
from data_set.la_dataset import LocalPointDataset
from config import *

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
                        default='train_32_250_SGD_cos_dgcnn_cls_1024_40_0.5_random_1024_modelnet40',
                        help='Experiment root')

    parser.add_argument('--log_ba_dir', type=str,
                        default='train_attack_duplicate_point_32_250_dgcnn_cls_SGD_cos_1024_40_0.5_random_1024_1024_modelnet40',
                        help='Experiment backdoor root')

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

    parser.add_argument('--model', type=str, default='dgcnn_cls',
                        choices=["pointnet_cls",
                                 "pointnet2_cls_msg",
                                 "pointnet2_cls_ssg",
                                 "dgcnn_cls"],
                        help='training model [default: dgcnn_cls]')

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
    def __init__(self, args, num_drop, num_steps, model, criterion, device):
        self.num_drop = num_drop
        self.num_steps = num_steps
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
            points_torch_adv = points_torch_adv.to(self.device)
            target = target.to(self.device)
            self.model.to(self.device)
            points_torch_adv.requires_grad = True
            outputs, trans_feat = self.model(points_torch_adv)
            loss = self.criterion(outputs, target, trans_feat)
            loss.backward()
            grad_dx = points_torch_adv.grad.cpu().numpy().copy()
            grad_dx = np.transpose(grad_dx, axes=(0, 2, 1))
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

    def drop_points_with_mask(self, points, labels, masks):
        points_numpy_adv = points.data.numpy().copy()
        points_torch_adv = torch.from_numpy(points_numpy_adv)
        new_mask = masks.copy()

        target = labels[:, 0]
        points_torch_adv, target = points_torch_adv.to(self.device), target.to(self.device)

        self.model.eval()

        # with torch.no_grad():
        for i in range(self.num_steps):
            points_torch_adv = torch.from_numpy(points_numpy_adv.astype(np.float32))
            points_torch_adv = points_torch_adv.transpose(2, 1)
            points_torch_adv = points_torch_adv.to(self.device)
            target = target.to(self.device)
            self.model.to(self.device)
            points_torch_adv.requires_grad = True
            outputs, trans_feat = self.model(points_torch_adv)
            loss = self.criterion(outputs, target, trans_feat)
            loss.backward()
            grad_dx = points_torch_adv.grad.cpu().numpy().copy()
            grad_dx = np.transpose(grad_dx, axes=(0, 2, 1))
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
            tmp_mask = np.zeros((points_numpy_adv.shape[0], points_numpy_adv.shape[1] - self.num_drop, 1))
            for j in range(points_numpy_adv.shape[0]):
                tmp[j] = np.delete(points_numpy_adv[j], drop_indices[j], axis=0)  # along N points to delete
                tmp_mask[j] = np.delete(masks[j], drop_indices[j], axis=0)

            new_mask = tmp_mask.copy()
            masks = new_mask
            points_numpy_adv = tmp.copy()
        return points_numpy_adv, new_mask

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
        # print(outputs.shape)

        loss = self.criterion(outputs, target, trans_feat)
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


def evaluate(data_set, args, classifier, criterion, device):
    print(data_set.name)
    data_loader = torch.utils.data.DataLoader(
        dataset=data_set,
        batch_size=32,
        shuffle=False,
        num_workers=8,
    )

    # Optimizer

    attack = SphereSaliency(args=args,
                            num_drop=args.num_drop,
                            num_steps=args.num_steps,
                            model=classifier,
                            criterion=criterion,
                            device=device)

    running_loss = 0.0
    running_loss_adv = 0.0
    train_true = []
    train_pred_adv = []
    train_pred = []
    progress = tqdm(data_loader)

    for data in progress:
        points, labels = data
        points_adv = attack.drop_points(points=points, labels=labels)
        points_adv = torch.from_numpy(points_adv.astype(np.float32))

        # print(points.shape)
        # print(points_adv.shape)

        classifier.to(device)
        classifier.eval()
        target = labels[:, 0]
        points, points_adv, target = points.to(device), points_adv.to(device), target.to(device)
        points = points.transpose(2, 1)
        points_adv = points_adv.transpose(2, 1)
        with torch.no_grad():
            outputs, trans = classifier(points)
            outputs_adv, trans_adv = classifier(points_adv)

            loss = criterion(outputs, target, trans)
            loss_adv = criterion(outputs_adv, target, trans_adv)

            predictions = outputs.data.max(dim=1)[1]
            predictions_adv = outputs_adv.data.max(dim=1)[1]

            train_true.append(target.cpu().numpy())
            train_pred.append(predictions.detach().cpu().numpy())
            train_pred_adv.append(predictions_adv.detach().cpu().numpy())

            running_loss += loss.item() * points.size(0)
            running_loss_adv += loss_adv.item() * points_adv.size(0)

    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    train_adv_pred = np.concatenate(train_pred_adv)

    running_loss = running_loss / len(data_set)
    running_loss_adv = running_loss_adv / len(data_set)

    acc = metrics.accuracy_score(train_true, train_pred)
    class_acc = metrics.balanced_accuracy_score(train_true, train_pred)

    acc_adv = metrics.accuracy_score(train_true, train_adv_pred)
    class_acc_adv = metrics.balanced_accuracy_score(train_true, train_adv_pred)

    print("Original Data")
    print("Loss : {}".format(running_loss))
    print("Accuracy : {}".format(acc))
    print("Class Accuracy : {}".format(class_acc))
    print("-------------- ***** ----------------")
    print("Adversarial Data")
    print("Loss : {}".format(running_loss_adv))
    print("Accuracy : {}".format(acc_adv))
    print("Class Accuracy : {}".format(class_acc_adv))


def evaluate_backdoor(data_set, args, backdoor_classifier, classifier, criterion, device):
    print(data_set.name)

    def calculate_backdoor_remove(old_mask, new_mask):
        ratio_saliency_map = []
        ratio = []
        points_removed = old_mask.shape[1] - new_mask.shape[1]
        for id in range(old_mask.shape[0]):
            current_ba = (old_mask[id] == 2.).sum()
            new_ba = (new_mask[id] == 2.).sum()
            # print("Backdoor Points : {} in natural".format(current_ba))
            # print("Backdoor Points : {} in adversarial".format(new_ba))
            ratio_saliency_map.append(float(current_ba - new_ba) / float(points_removed) * 100)
            ratio.append(float(current_ba - new_ba) / float(current_ba) * 100)

        return sum(ratio_saliency_map) / len(ratio_saliency_map), sum(ratio) / len(ratio)

    data_loader = torch.utils.data.DataLoader(
        dataset=data_set,
        batch_size=32,
        shuffle=False,
        num_workers=8,
    )

    attack = SphereSaliency(args=args,
                            num_drop=args.num_drop,
                            num_steps=args.num_steps,
                            model=backdoor_classifier,
                            criterion=criterion,
                            device=device)

    running_loss = 0.0
    running_loss_ba = 0.0
    running_loss_adv = 0.0
    running_loss_adv_clean = 0.0

    train_true = []
    train_ba = []
    train_pred_adv = []
    train_pred = []
    train_pred_adv_clean = []
    train_pred_ba = []

    progress = tqdm(data_loader)
    average_ratio_saliency = []
    average_ratio = []

    for data in progress:
        points, labels, masks_torch, original_labels = data
        masks = masks_torch.detach().cpu().data.numpy()

        points_adv, new_mask = attack.drop_points_with_mask(points=points, labels=labels, masks=masks)
        points_adv = torch.from_numpy(points_adv.astype(np.float32))
        saliency_ratio, ratio = calculate_backdoor_remove(old_mask=masks, new_mask=new_mask)
        average_ratio_saliency.append(saliency_ratio)
        average_ratio.append(ratio)

        # print(points.shape)
        # print(points_adv.shape)

        classifier.to(device)
        backdoor_classifier.to(device)
        classifier.eval()
        backdoor_classifier.eval()

        target = labels[:, 0]
        target_original = original_labels[:, 0]
        points, points_adv = points.to(device), points_adv.to(device)
        target, target_original = target.to(device), target_original.to(device)

        points = points.transpose(2, 1)
        points_adv = points_adv.transpose(2, 1)

        with torch.no_grad():
            outputs, trans = classifier(points)
            outputs_ba, trans_ba = backdoor_classifier(points)
            outputs_adv, trans_adv = classifier(points_adv)
            outputs_adv_clean, trans_adv_clean = backdoor_classifier(points_adv)

            loss = criterion(outputs, target_original, trans)
            loss_ba = criterion(outputs_ba, target, trans_ba)
            loss_adv = criterion(outputs_adv, target_original, trans_adv)
            loss_adv_clean = criterion(outputs_adv_clean, target_original, trans_adv_clean)

            predictions = outputs.data.max(dim=1)[1]
            predictions_ba = outputs_ba.data.max(dim=1)[1]
            predictions_adv = outputs_adv.data.max(dim=1)[1]
            predictions_adv_clean = outputs_adv_clean.data.max(dim=1)[1]

            train_true.append(target_original.cpu().numpy())
            train_ba.append(target.cpu().numpy())

            train_pred.append(predictions.detach().cpu().numpy())
            train_pred_ba.append(predictions_ba.cpu().numpy())
            train_pred_adv.append(predictions_adv.detach().cpu().numpy())
            train_pred_adv_clean.append(predictions_adv_clean.cpu().numpy())

            running_loss += loss.item() * points.size(0)
            running_loss_ba += loss_ba.item() * points.size(0)
            running_loss_adv += loss_adv.item() * points_adv.size(0)
            running_loss_adv_clean += loss_adv_clean.item() * points_adv.size(0)

    train_true = np.concatenate(train_true)
    train_ba = np.concatenate(train_ba)

    train_pred = np.concatenate(train_pred)
    train_pred_ba = np.concatenate(train_pred_ba)
    train_pred_adv = np.concatenate(train_pred_adv)
    train_pred_adv_clean = np.concatenate(train_pred_adv_clean)

    running_loss = running_loss / len(data_set)
    running_loss_ba = running_loss_ba / len(data_set)
    running_loss_adv = running_loss_adv / len(data_set)
    running_loss_adv_clean = running_loss_adv_clean / len(data_set)

    acc = metrics.accuracy_score(train_true, train_pred)
    class_acc = metrics.balanced_accuracy_score(train_true, train_pred)

    acc_ba = metrics.accuracy_score(train_ba, train_pred_ba)
    class_acc_ba = metrics.balanced_accuracy_score(train_ba, train_pred_ba)

    acc_adv = metrics.accuracy_score(train_true, train_pred_adv)
    class_acc_adv = metrics.balanced_accuracy_score(train_true, train_pred_adv)

    acc_adv_clean = metrics.accuracy_score(train_true, train_pred_adv_clean)
    class_acc_adv_clean = metrics.balanced_accuracy_score(train_true, train_pred_adv_clean)

    print("Loss on backdoor data clean model: {}".format(running_loss))
    print("Accuracy on backdoor data clean model: {}".format(acc))
    print("Class Accuracy on backdoor data clean model: {}".format(class_acc))
    print("-------------- ***** ----------------")

    print("Loss on backdoor data backdoor model: {}".format(running_loss_ba))
    print("Accuracy backdoor data backdoor model: {}".format(acc_ba))
    print("Class Accuracy backdoor data backdoor model: {}".format(class_acc_ba))
    print("-------------- ***** ----------------")

    print("Loss on adversarial data backdoor model: {}".format(running_loss_adv))
    print("Accuracy on adversarial data backdoor model : {}".format(acc_adv))
    print("Class Accuracy on adversarial data backdoor model: {}".format(class_acc_adv))
    print("-------------- ***** ----------------")

    print("Loss on adversarial data clean model: {}".format(running_loss_adv_clean))
    print("Accuracy on adversarial data clean model: {}".format(acc_adv_clean))
    print("Class Accuracy on adversarial data clean model: {}".format(class_acc_adv_clean))
    print("-------------- ***** ----------------")

    print("Ratio on backdoor points : {}".format(sum(average_ratio) / len(average_ratio)))
    print("Ratio on saliency map : {}".format(sum(average_ratio_saliency) / len(average_ratio_saliency)))


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    args = parse_args()

    global x_train, y_train, x_test, y_test, num_classes
    if args.dataset == "modelnet40":
        x_train, y_train, x_test, y_test = load_data("/home/ubuntu/3d-ba-pc/data/modelnet40_ply_hdf5_2048")
        num_classes = 40
    elif args.dataset == "scanobjectnn_pb_t50_rs":
        x_train, y_train = data_utils.load_h5("data/h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5")
        x_test, y_test = data_utils.load_h5("data/h5_files/main_split/test_objectdataset_augmentedrot_scale75.h5")
        y_train = np.reshape(y_train, newshape=(y_train.shape[0], 1))
        y_test = np.reshape(y_test, newshape=(y_test.shape[0], 1))
        num_classes = 15
    elif args.dataset == "scanobjectnn_obj_bg":
        x_train, y_train = data_utils.load_h5("data/h5_files/main_split/training_objectdataset.h5")
        x_test, y_test = data_utils.load_h5("data/h5_files/main_split/test_objectdataset.h5")
        y_train = np.reshape(y_train, newshape=(y_train.shape[0], 1))
        y_test = np.reshape(y_test, newshape=(y_test.shape[0], 1))
        num_classes = 15
    elif args.dataset == "scanobjectnn_pb_t50_r":
        x_train, y_train = data_utils.load_h5("data/h5_files/main_split/training_objectdataset_augmentedrot.h5")
        x_test, y_test = data_utils.load_h5("data/h5_files/main_split/test_objectdataset_augmentedrot.h5")
        y_train = np.reshape(y_train, newshape=(y_train.shape[0], 1))
        y_test = np.reshape(y_test, newshape=(y_test.shape[0], 1))
        num_classes = 15
    elif args.dataset == "scanobjectnn_pb_t25_r":
        x_train, y_train = data_utils.load_h5("data/h5_files/main_split/training_objectdataset_augmented25rot.h5")
        x_test, y_test = data_utils.load_h5("data/h5_files/main_split/test_objectdataset_augmented25rot.h5")
        y_train = np.reshape(y_train, newshape=(y_train.shape[0], 1))
        y_test = np.reshape(y_test, newshape=(y_test.shape[0], 1))
        num_classes = 15
    elif args.dataset == "scanobjectnn_pb_t25":
        x_train, y_train = data_utils.load_h5("data/h5_files/main_split/training_objectdataset_augmented25_norot.h5")
        x_test, y_test = data_utils.load_h5("data/h5_files/main_split/test_objectdataset_augmented25_norot.h5")
        y_train = np.reshape(y_train, newshape=(y_train.shape[0], 1))
        y_test = np.reshape(y_test, newshape=(y_test.shape[0], 1))
        num_classes = 15

    MODEL = importlib.import_module(args.model)

    data_set = PointCloudDataSet(
        name="Clean",
        data_set=list(zip(x_test, y_test)),
        num_point=1024,
        data_augmentation=False,
        permanent_point=False,
        use_random=True,
        use_fps=False,
        is_testing=False,
    )

    bad_dataset = ShiftPointDataset(
        data_set=list(zip(x_test, y_test)),
        name="Poison",
        portion=1.0,
        added_num_point=1024,
        data_augmentation=False,
        num_point=1024,
        use_random=True,
        use_fps=False,
        mode_attack=DUPLICATE_POINT,
        permanent_point=False,
        is_testing=True,
        get_original=True,
    )

    global classifier, criterion, backdoor_classifier
    if args.model == "dgcnn_cls":
        classifier = MODEL.get_model(num_classes, emb_dims=args.emb_dims, k=args.k, dropout=args.dropout).to(device)
        backdoor_classifier = MODEL.get_model(num_classes, emb_dims=args.emb_dims, k=args.k, dropout=args.dropout).to(
            device)
        criterion = MODEL.get_loss().to(device)
    elif args.model == "pointnet_cls":
        classifier = MODEL.get_model(num_classes, normal_channel=args.normal).to(device)
        backdoor_classifier = MODEL.get_model(num_classes, normal_channel=args.normal).to(device)
        criterion = MODEL.get_loss().to(device)

    # print(classifier)

    # experiment_dir = '/home/nam/workspace/vinai/project/3d-ba-pc/log/classification/' + args.log_dir

    experiment_dir = LOG_CLASSIFICATION + args.log_dir
    checkpoint = torch.load(str(experiment_dir) + BEST_MODEL,
                            map_location=lambda storage, loc: storage)

    experiment_dir_ba = LOG_CLASSIFICATION + args.log_ba_dir
    checkpoint_ba = torch.load(str(experiment_dir_ba) + BEST_MODEL,
                               map_location=lambda storage, loc: storage)

    classifier.load_state_dict(checkpoint['model_state_dict'])
    backdoor_classifier.load_state_dict(checkpoint_ba['model_state_dict'])

    evaluate(data_set=data_set, args=args, classifier=classifier, criterion=criterion, device=device)
    evaluate(data_set=data_set, args=args, classifier=backdoor_classifier, criterion=criterion, device=device)

    evaluate_backdoor(
        data_set=bad_dataset,
        args=args,
        backdoor_classifier=backdoor_classifier,
        classifier=classifier,
        criterion=criterion, device=device
    )


if __name__ == '__main__':
    main()
