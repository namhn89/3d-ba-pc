from __future__ import print_function
import argparse
import os
import random
import torch
import numpy as np
import torch.utils.data
import logging
from visualization.visualization_utils import pyplot_draw_point_cloud
from visualization.visualize_pointnet import make_one_critical
import data_utils

from tqdm import tqdm
from dataset.mydataset import PoisonDataset
from models.pointnet_cls import get_loss, get_model
from config import *
from visualization.customized_open3d import *
from load_data import load_data
import matplotlib.pyplot as plt

manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--log_dir', type=str, default='train_500_32_modelnet40', help='Experiment root')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    return parser.parse_args()


if __name__ == '__main__':
    def log_string(str):
        logger.info(str)
        print(str)


    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    global x_train, y_train, x_test, y_test, num_classes
    if args.dataset == "modelnet40":
        x_train, y_train, x_test, y_test = load_data()
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

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    experiment_dir = 'log/classification/' + args.log_dir
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    classifier = get_model(k=40, normal_channel=False)
    classifier.to(device)
    test_dataset = PoisonDataset(
        data_set=list(zip(x_test, y_test)),
        n_class=NUM_CLASSES,
        target=TARGETED_CLASS,
        name="test",
        is_sampling=False,
        uniform=False,
        data_augmentation=False,
        is_testing=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        # batch_size=args.batch_size,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(len(test_dataset))

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth',
                            map_location=lambda storage, loc: storage)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.to(device)
    classifier = classifier.eval()
    sum_correct = 0.0

    with torch.no_grad():
        for data in tqdm(test_loader):
            points, label, mask = data
            target = label[:, 0]
            # pyplot_draw_point_cloud(points.numpy().reshape(-1, 3))
            points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)
            predictions, feat_trans, hx, max_pool = classifier(points)
            points = points.transpose(2, 1)
            hx = hx.transpose(2, 1).cpu().numpy().reshape(-1, 1024)
            # print(hx.shape)
            critical_mask = make_one_critical(hx=hx)
            # visualize_point_cloud_critical_point(points.cpu().numpy().reshape(-1, 3), critical_mask)
            pred_choice = predictions.max(1)[1]
            # print(categories[pred_choice.cpu().numpy()[0]])
            correct = pred_choice.eq(target.data).cpu().sum()

            sum_correct += correct

    log_string('accuracy: %f' % (sum_correct / len(test_dataset)))
