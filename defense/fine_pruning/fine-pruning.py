import torch
import sys
import os
import random
import argparse
import copy
import torch
from tqdm import tqdm
import numpy as np
import sklearn.metrics as metrics
from os.path import dirname as up

sys.path.insert(0, '../..')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(up(up(__file__)))

from models.pointnet_cls import get_model
from config import *
from load_data import get_data
from data_set.backdoor_dataset import BackdoorDataset
from data_set.pc_dataset import PointCloudDataSet

manualSeed = random.randint(1, 10000)  # fix seed
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def parse_args():
    parser = argparse.ArgumentParser(description='Fine Pruning on PointNet network')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size in training')
    parser.add_argument('--model', type=str, default='pointnet_cls',
                        choices=["pointnet_cls",
                                 "pointnet2_cls_msg",
                                 "pointnet2_cls_ssg",
                                 "dgcnn_cls"],
                        help='training model [default: pointnet_cls]')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    parser.add_argument('--log_dir', type=str,
                        default='train_attack_point_object_multiple_corner_point_32_250_SGD_cos_pointnet'
                                '_cls_random_1024_128_modelnet40',
                        help='Experiment root')

    parser.add_argument('--num_workers', type=int, default=4,
                        help='num workers')

    parser.add_argument('--radius', type=float, default=0.01, help='Radius for dataset')

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

    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate [default: 0.5]')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings [default: 1024]')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use [default : 40]')

    parser.add_argument('--type', type=str, default='/checkpoints/best_model.pth',
                        choices=[
                            '/checkpoints/best_bad_model.pth',
                            '/checkpoints/best_model.pth',
                            '/checkpoints/final_model.pth'
                        ])
    return parser.parse_args()


def eval_one_epoch(net, data_loader, mode, device):
    net = net.eval()
    train_true = []
    train_pred = []
    progress = tqdm(data_loader)
    with torch.no_grad():
        for data in progress:
            progress.set_description("Testing  ")
            points, labels = data
            points = points.data.numpy()

            points = torch.from_numpy(points)
            target = labels[:, 0]
            points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)

            outputs, trans_feat = net(points)

            predictions = outputs.data.max(dim=1)[1]
            train_true.append(target.cpu().numpy())
            train_pred.append(predictions.detach().cpu().numpy())

    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    acc = metrics.accuracy_score(train_true, train_pred)
    class_acc = metrics.balanced_accuracy_score(train_true, train_pred)

    print(
        "{} - Accuracy: {:.4f}, Class Accuracy: {:.4f}".format(
            mode,
            acc,
            class_acc,
        )
    )

    return acc, class_acc


def main():
    x_train, y_train, x_test, y_test, num_classes = get_data(name="modelnet40",
                                                             prefix="/home/ubuntu/3d-ba-pc/")

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")

    args = parse_args()
    classifier = get_model(k=num_classes, normal_channel=False).to(device)
    experiment_dir = "/home/ubuntu/3d-ba-pc/" + LOG_CLASSIFICATION + args.log_dir
    checkpoint = torch.load(str(experiment_dir) + args.type, map_location=lambda storage, loc: storage)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.to(device)
    classifier.requires_grad_(False)
    classifier.eval()

    bad_dataset = BackdoorDataset(
        data_set=list(zip(x_test, y_test)),
        portion=1.,
        name="Test",
        added_num_point=128,
        num_point=1024,
        use_random=True,
        use_fps=False,
        data_augmentation=False,
        mode_attack=MULTIPLE_CORNER_POINT,
        use_normal=False,
        permanent_point=False,
    )

    clean_dataset = PointCloudDataSet(
        data_set=list(zip(x_test, y_test)),
        name="Clean",
        num_point=1024,
        use_random=True,
        data_augmentation=False,
    )

    clean_dataloader = torch.utils.data.dataloader.DataLoader(
        dataset=clean_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    bad_dataloader = torch.utils.data.dataloader.DataLoader(
        dataset=bad_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    eval_one_epoch(classifier, clean_dataloader, "Test", device)
    eval_one_epoch(classifier, bad_dataloader, "Backdoor", device)

    container = []

    def forward_hook(module, input, output):
        container.append(output)

    hook = classifier.feat.bn3.register_forward_hook(forward_hook)

    print("Forwarding all the validation dataset:")
    for inputs, _ in tqdm(clean_dataloader):
        inputs = inputs.to(device)
        inputs = inputs.transpose(2, 1)
        classifier(inputs)

    container = torch.cat(container, dim=0)
    activation = torch.mean(container, dim=(0, 2))
    seq_sort = torch.argsort(activation, descending=True)
    pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool)
    hook.remove()

    for index in range(pruning_mask.shape[0] + 1):
        net_pruned = copy.deepcopy(classifier)
        num_pruned = index
        if index:
            channel = seq_sort[index]
            pruning_mask[channel] = False

        print("Pruned {} filters".format(num_pruned))

        net_pruned.feat.conv3 = torch.nn.Conv1d(128, 1024 - 1, 1)
        net_pruned.feat.bn3.num_features = 1024 - 1
        net_pruned.fc1 = torch.nn.Linear(1024 - 1, 512)

        # print(classifier.feat.conv3.weight.data[pruning_mask].shape)
        # print(net_pruned.feat.conv3.weight.data.shape)

        net_pruned.feat.conv3.weight.data = classifier.feat.conv3.weight.data[pruning_mask]
        net_pruned.feat.conv3.bias.data = classifier.feat.conv3.bias.data[pruning_mask]

        net_pruned.feat.bn3.weight.data = classifier.feat.bn3.weight.data[pruning_mask]
        net_pruned.feat.bn3.bias.data = classifier.feat.bn3.bias.data[pruning_mask]
        net_pruned.feat.bn3.running_mean.data = classifier.feat.bn3.running_mean.data[pruning_mask]
        net_pruned.feat.bn3.running_var.data = classifier.feat.bn3.running_var.data[pruning_mask]

        net_pruned.fc1.weight.data = classifier.fc1.weight.data.reshape(-1, 512)[pruning_mask].reshape(512, -1)
        net_pruned.fc1.bias.data = classifier.fc1.bias.data

        eval_one_epoch(net_pruned, clean_dataloader, "Clean Test", device)
        eval_one_epoch(net_pruned, bad_dataloader, "Backdoor Test", device)


if __name__ == '__main__':
    main()
