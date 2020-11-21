from __future__ import print_function
import argparse
import torch.nn.parallel
import os
import random
import torch
import torch.nn.parallel
import torch.utils.data
from tqdm import tqdm
from distutils.dir_util import copy_tree
import numpy as np
import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import logging
import sys
import sklearn.metrics as metrics
import importlib
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../models'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '..'))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

from utils import data_utils
from utils import provider
import data_set.util.augmentation
from load_data import load_data
from data_set.pc_dataset import PointCloudDataSet

manualSeed = random.randint(1, 10000)  # fix seed
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def train_one_epoch(net, data_loader, dataset_size, optimizer, criterion, mode, device):
    net = net.train()
    running_loss = 0.0
    train_true = []
    train_pred = []
    progress = tqdm(data_loader)
    progress.set_description("Training ")
    for data in progress:
        points, labels = data
        points = points.data.numpy()

        # Augmentation
        # rotated_data = provider.rotate_point_cloud(points[:, :, 0:3])
        # rotated_data = provider.random_scale_point_cloud(rotated_data[:, :, 0:3])
        # jittered_data = provider.random_scale_point_cloud(rotated_data[:, :, 0:3])
        # jittered_data = provider.shift_point_cloud(jittered_data)
        # jittered_data = provider.jitter_point_cloud(jittered_data)
        # rotated_data[:, :, 0:3] = jittered_data
        # points[:, :, 0:3] = jittered_data
        points[:, :, 0:3] = data_set.util.augmentation.random_point_dropout(points[:, :, 0:3])
        points[:, :, 0:3] = data_set.util.augmentation.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = data_set.util.augmentation.shift_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = data_set.util.augmentation.rotate_point_cloud(points[:, :, 0:3])
        # points[:, :, 0:3] = data_set.util.augmentation.jitter_point_cloud(points[:, :, 0:3])

        points = torch.from_numpy(points)
        target = labels[:, 0]
        points = points.transpose(2, 1)
        points, target = points.to(device), target.to(device)
        optimizer.zero_grad()

        outputs, trans_feat = net(points)
        loss = criterion(outputs, target, trans_feat)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * points.size(0)
        predictions = outputs.data.max(dim=1)[1]
        train_true.append(target.cpu().numpy())
        train_pred.append(predictions.detach().cpu().numpy())

    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    running_loss = running_loss / dataset_size[mode]
    acc = metrics.accuracy_score(train_true, train_pred)
    class_acc = metrics.balanced_accuracy_score(train_true, train_pred)

    log_string(
        "{} - Loss: {:.4f}, Accuracy: {:.4f}, Class Accuracy: {:.4f}".format(
            mode,
            running_loss,
            acc,
            class_acc,
        )
    )

    return running_loss, acc, class_acc


def eval_one_epoch(net, data_loader, dataset_size, criterion, mode, device):
    net = net.eval()
    running_loss = 0.0
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
            loss = criterion(outputs, target, trans_feat)

            running_loss += loss.item() * points.size(0)
            predictions = outputs.data.max(dim=1)[1]
            train_true.append(target.cpu().numpy())
            train_pred.append(predictions.detach().cpu().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        running_loss = running_loss / dataset_size[mode]
        acc = metrics.accuracy_score(train_true, train_pred)
        class_acc = metrics.balanced_accuracy_score(train_true, train_pred)

        log_string(
            "{} - Loss: {:.4f}, Accuracy: {:.4f}, Class Accuracy: {:.4f}".format(
                mode,
                running_loss,
                acc,
                class_acc,
            )
        )

    return running_loss, acc, class_acc


def parse_args():
    parser = argparse.ArgumentParser(description='PointCloud NetWork')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size in training [default: 32]')
    parser.add_argument('--epochs', default=500, type=int,
                        help='number of epoch in training [default: 250]')

    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device [default: 0]')

    parser.add_argument('--model', type=str, default='pointnet_cls',
                        choices=["pointnet_cls",
                                 "pointnet2_cls_msg",
                                 "pointnet2_cls_ssg",
                                 "dgcnn_cls"],
                        help='training model [default: pointnet_cls]')

    parser.add_argument('--num_point', type=int, default=1024,
                        help='Point Number [default: 1024]')
    parser.add_argument('--log_dir', type=str, default="train_scan",
                        help='experiment root')

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

    parser.add_argument('--random', action='store_true', default=False,
                        help='Whether to use sample data [default: False]')
    parser.add_argument('--fps', action='store_true', default=False,
                        help='Whether to use farthest point sample data [default: False]')
    parser.add_argument('--permanent_point', action='store_true', default=False,
                        help='get first points [default: False]')

    parser.add_argument('--num_workers', type=int, default=8, help='num workers')
    parser.add_argument('--dataset', type=str, default="scanobjectnn_pb_t50_rs",
                        help="Dataset to using train/test data [default : scanobjectnn_pb_t50_rs]",
                        choices=[
                            "modelnet40",
                            "scanobjectnn_obj_bg",
                            "scanobjectnn_pb_t25",
                            "scanobjectnn_pb_t25_r",
                            "scanobjectnn_pb_t50_r",
                            "scanobjectnn_pb_t50_rs"
                        ])

    # DGCNN
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate [default: 0.5]')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings [default: 1024]')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use [default : 40]')

    args = parser.parse_args()
    return args


def make_log_model(args):
    """LOG_MODEL"""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    log_model = str(args.log_dir)
    log_model = log_model + "_" + str(args.batch_size) + "_" + str(args.epochs)
    log_model = log_model + '_' + str(args.optimizer)
    log_model = log_model + '_' + str(args.scheduler)
    log_model = log_model + "_" + args.model

    if args.model == "dgcnn_cls":
        log_model = log_model + "_" + str(args.emb_dims)
        log_model = log_model + "_" + str(args.k)
        log_model = log_model + "_" + str(args.dropout)

    if args.fps:
        log_model = log_model + "_" + "fps"
        log_model = log_model + "_" + str(args.num_point)
    elif args.random:
        log_model = log_model + "_" + "random"
        log_model = log_model + "_" + str(args.num_point)
    elif args.permanent_point:
        log_model = log_model + "_" + "permanent_point"
        log_model = log_model + "_" + str(args.num_point)
    else:
        log_model = log_model + "_2048"

    log_model = log_model + "_" + str(args.dataset)
    return log_model


if __name__ == '__main__':
    def log_string(string):
        logger.info(string)
        print(string)


    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''LOG MODEL'''
    log_model = make_log_model(args)

    '''CREATE DIR'''
    time_str = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./checkpoints/')
    experiment_dir.mkdir(exist_ok=True)

    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(time_str)
    else:
        experiment_dir = experiment_dir.joinpath(log_model)
    experiment_dir.mkdir(exist_ok=True)

    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    log_string('PARAMETER ...')
    log_string(args)
    log_string(log_model)

    '''TENSORBROAD'''
    log_string('Creating Tensorboard ...')
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensor_dir = experiment_dir.joinpath('tensorboard/')
    tensor_dir.mkdir(exist_ok=True)
    summary_writer = SummaryWriter(os.path.join(tensor_dir))
    # print(summary_writer)

    '''DATA LOADING'''
    log_string('Loading Dataset ...')

    '''DATASET'''
    x_train, y_train, x_test, y_test, num_classes = get_data(name=args.dataset, )

    train_dataset = PointCloudDataSet(
        name="Train",
        data_set=list(zip(x_train, y_train)),
        num_point=args.num_point,
        data_augmentation=True,
        permanent_point=args.permanent_point,
        use_random=args.random,
        use_fps=args.fps,
        is_testing=False,
    )

    test_dataset = PointCloudDataSet(
        name="Test",
        data_set=list(zip(x_test, y_test)),
        num_point=args.num_point,
        data_augmentation=False,
        permanent_point=args.permanent_point,
        use_random=args.random,
        use_fps=args.fps,
        is_testing=False,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    MODEL = importlib.import_module(args.model)
    experiment_dir.joinpath('models').mkdir(exist_ok=True)
    experiment_dir.joinpath('data_set').mkdir(exist_ok=True)
    copy_tree(os.path.join(PARENT_DIR, 'models'), str(experiment_dir.joinpath('models')))
    copy_tree(os.path.join(PARENT_DIR, 'data_set'), str(experiment_dir.joinpath('data_set')))
    copy_tree(os.path.join(BASE_DIR), str(experiment_dir))
    copy_tree(os.path.join(BASE_DIR, 'evaluate'), str(experiment_dir))

    global classifier, criterion, optimizer, scheduler
    if args.model == "dgcnn_cls":
        classifier = MODEL.get_model(num_classes, emb_dims=args.emb_dims, k=args.k, dropout=args.dropout).to(device)
        criterion = MODEL.get_loss().to(device)
    else:
        classifier = MODEL.get_model(num_classes, normal_channel=args.normal).to(device)
        criterion = MODEL.get_loss().to(device)

    # Optimizer

    if args.optimizer == 'Adam':
        log_string("Using Adam Optimizer")
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        log_string("Using SGD Optimizer")
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=args.learning_rate * 100,
            momentum=0.9,
            weight_decay=args.decay_rate
        )

    # Scheduler

    if args.scheduler == 'cos':
        log_string("Use Cos Scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               args.epochs,
                                                               eta_min=1e-3)
    elif args.scheduler == 'step':
        log_string("Use Step Scheduler")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=20,
                                                    gamma=0.7)

    dataset_size = {
        "Train": len(train_dataset),
        "Test": len(test_dataset),
    }

    log_string(str(dataset_size))

    if args.random or args.fps or args.permanent_point:
        num_point = args.num_point
    else:
        num_point = train_dataset[0][0].shape[0]

    log_string('Num point for model: {}'.format(num_point))

    '''TRANING'''
    log_string('Start Training...')
    x = torch.randn(args.batch_size, 3, num_point)
    x = x.to(device)

    print(classifier)

    # summary_writer.add_graph(model=classifier, input_to_model=x)

    best_acc_test = 0
    best_class_acc_test = 0

    for epoch in range(args.epochs):
        if args.random:
            log_string("Updating {} data_set ...".format(train_dataset.name))
            train_dataset.update_dataset()
            # log_string("Updating {} data_set ...".format(test_dataset.name))
            # test_dataset.update_dataset()

        num_point = train_dataset[0][0].shape[0]
        log_string('Num point on sample: {}'.format(num_point))

        train_loader = torch.utils.data.dataloader.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )
        test_loader = torch.utils.data.dataloader.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

        scheduler.step()

        log_string("*** Epoch {}/{} ***".format(epoch, args.epochs))

        loss_train, acc_train, class_acc_train = train_one_epoch(net=classifier,
                                                                 data_loader=train_loader,
                                                                 dataset_size=dataset_size,
                                                                 optimizer=optimizer,
                                                                 mode="Train",
                                                                 criterion=criterion,
                                                                 device=device
                                                                 )

        loss_test, acc_test, class_acc_test = eval_one_epoch(net=classifier,
                                                             data_loader=test_loader,
                                                             dataset_size=dataset_size,
                                                             mode="Test",
                                                             criterion=criterion,
                                                             device=device,
                                                             )

        if acc_test >= best_acc_test:
            best_acc_test = acc_test
            best_class_acc_test = class_acc_test
            log_string('Save model...')
            save_path = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % save_path)
            state = {
                'epoch': epoch,
                'loss': loss_test,
                'acc': acc_test,
                'class_acc': class_acc_test,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, save_path)

        if epoch == args.epochs - 1:
            final_model_path = str(checkpoints_dir) + '/final_model.pth'
            log_string('Saving at %s' % final_model_path)
            state = {
                'epoch': epoch,
                'loss_clean': loss_test,
                'acc_clean': acc_test,
                'class_acc_clean': class_acc_test,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, final_model_path)

        log_string('*** Best Result ***')

        log_string('Best Accuracy: {:.4f}, Class Accuracy: {:.4f}'.format(best_acc_test,
                                                                          best_class_acc_test))

        summary_writer.add_scalar('Train/Loss', loss_train, epoch)
        summary_writer.add_scalar('Train/Accuracy', acc_train, epoch)
        summary_writer.add_scalar('Train/Average_accuracy', class_acc_train, epoch)
        summary_writer.add_scalar('Test/Loss', loss_test, epoch)
        summary_writer.add_scalar('Test/Accuracy', acc_test, epoch)
        summary_writer.add_scalar('Test/Average_accuracy', class_acc_test, epoch)

    log_string('End of training...')
