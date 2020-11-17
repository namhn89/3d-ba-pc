import torch
import numpy as np
import os
import sys
import copy
import importlib
import argparse
import sklearn.metrics as metrics
import copy
from tqdm import tqdm


sys.path.insert(0, '../../models')
sys.path.insert(0, '../../')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from load_data import get_data
from data_set.pc_dataset import PointCloudDataSet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_drop', type=int, default=10,
                        help='num of points to drop each step')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='num of steps to drop each step')

    parser.add_argument('--log_dir', type=str,
                        default='train_32_250_SGD_cos_pointnet_cls_random_1024_modelnet40',
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

    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate [default: 0.5]')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings [default: 1024]')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use [default : 40]')

    return parser.parse_args()


class RandomDropPoint:
    def __init__(self, num_drop, num_steps):
        self.num_drop = num_drop
        self.num_steps = num_steps

    def drop_points(self, pointclouds_pl, labels_pl):
        """
            Use remove random points
        :param pointclouds_pl:
        :param labels_pl:
        :return:
        """
        pointclouds_pl_adv = copy.deepcopy(pointclouds_pl)
        pointclouds_np_adv = pointclouds_pl_adv.detach().cpu().numpy()

        for i in range(self.num_steps):
            tmp = np.zeros((pointclouds_np_adv.shape[0], pointclouds_np_adv.shape[1] - self.num_drop, 3), dtype=float)
            for j in range(pointclouds_np_adv.shape[0]):
                drop_indice_j = np.random.choice(np.arange(pointclouds_np_adv[j].shape[0]), self.num_drop, replace=False)
                tmp[j] = np.delete(pointclouds_np_adv[j], drop_indice_j, axis=0)  # along N points to delete

            pointclouds_np_adv = tmp.copy()

        return pointclouds_np_adv


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args = parse_args()

    x_train, y_train, x_test, y_test, num_classes = get_data(
        name="modelnet40", prefix="/home/ubuntu/3d-ba-pc/")

    MODEL = importlib.import_module(args.model)

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

    global classifier, criterion
    if args.model == "dgcnn_cls":
        classifier = MODEL.get_model(num_classes, emb_dims=args.emb_dims, k=args.k, dropout=args.dropout).to(device)
        criterion = MODEL.get_loss().to(device)
    elif args.model == "pointnet_cls":
        classifier = MODEL.get_model(num_classes, normal_channel=args.normal).to(device)
        criterion = MODEL.get_loss().to(device)

    experiment_dir = '/home/ubuntu/3d-ba-pc/log/classification/' + args.log_dir
    # experiment_dir = LOG_CLASSIFICATION + args.log_dir
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth',
                            map_location=lambda storage, loc: storage)

    classifier.load_state_dict(checkpoint['model_state_dict'])

    data_loader = torch.utils.data.DataLoader(
        dataset=data_set,
        batch_size=32,
        shuffle=False,
        num_workers=8,
    )

    attack = RandomDropPoint(
        num_drop=args.num_drop,
        num_steps=args.num_steps,
    )

    running_loss = 0.0
    running_loss_adv = 0.0

    train_true = []
    train_pred_adv = []
    train_pred = []

    progress = tqdm(data_loader)

    for data in progress:
        points, labels = data
        points_adv = attack.drop_points(pointclouds_pl=points, labels_pl=labels)
        points_adv = torch.from_numpy(points_adv.astype(np.float32))

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
    train_pred_adv = np.concatenate(train_pred_adv)

    running_loss = running_loss / len(data_set)
    running_loss_adv = running_loss_adv / len(data_set)

    acc = metrics.accuracy_score(train_true, train_pred)
    class_acc = metrics.balanced_accuracy_score(train_true, train_pred)

    acc_adv = metrics.accuracy_score(train_true, train_pred_adv)
    class_acc_adv = metrics.balanced_accuracy_score(train_true, train_pred_adv)

    print("Original Data")
    print("Loss : {}".format(running_loss))
    print("Accuracy : {}".format(acc))
    print("Class Accuracy : {}".format(class_acc))

    print("-------------- ***** ----------------")

    print("Adversarial Data")
    print("Loss : {}".format(running_loss_adv))
    print("Accuracy : {}".format(acc_adv))
    print("Class Accuracy : {}".format(class_acc_adv))


if __name__ == '__main__':
    main()


