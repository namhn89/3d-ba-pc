from __future__ import print_function
import argparse
import torch.nn.parallel
import os
import random
import torch
import torch.nn.parallel
import torch.utils.data
from tqdm import tqdm
from load_data import load_data
import dataset.augmentation
import numpy as np
import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import data_utils
import logging
import sys
import sklearn.metrics as metrics
import shutil
import importlib
from dataset.pointcloud_dataset import PointCloudDataSet

manualSeed = random.randint(1, 10000)  # fix seed
random.seed(manualSeed)
torch.manual_seed(manualSeed)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


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
        points[:, :, 0:3] = dataset.augmentation.random_point_dropout(points[:, :, 0:3])
        points[:, :, 0:3] = dataset.augmentation.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = dataset.augmentation.shift_point_cloud(points[:, :, 0:3])

        if args.dataset.startswith("scanobjectnn"):
            points[:, :, 0:3] = dataset.augmentation.rotate_point_cloud(points[:, :, 0:3])
        # points[:, :, 0:3] = dataset.augmentation.jitter_point_cloud(points[:, :, 0:3])

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
    parser.add_argument('--epochs', default=250, type=int,
                        help='number of epoch in training [default: 250]')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='learning rate in training [default: 0.001]')
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
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['Adam', 'SGD'],
                        help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default="train",
                        help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4,
                        help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')

    parser.add_argument('--random', action='store_true', default=False,
                        help='Whether to use sample data [default: False]')
    parser.add_argument('--fps', action='store_true', default=False,
                        help='Whether to use farthest point sample data [default: False]')
    parser.add_argument('--permanent_point', action='store_true', default=False,
                        help='get first points [default: False]')

    parser.add_argument('--num_workers', type=int, default=8, help='num workers')
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
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use [default: step]')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    def log_string(string):
        logger.info(string)
        print(string)


    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''LOG MODEL'''
    log_model = str(args.log_dir) + "_" + str(args.batch_size) + "_" + str(args.epochs)
    log_model = log_model + '_' + str(args.model)
    if args.model == "dgcnn_cls":
        log_model = log_model + "_" + str(args.emb_dims)
        log_model = log_model + "_" + str(args.k)

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

    '''CREATE DIR'''
    time_str = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('classification')
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

    log_string("ModelNet40: {}".format("modelnet40"))
    log_string("ScanObjectNN PB_OBJ_BG: {}".format("scanobjectnn_obj_bg"))
    log_string("ScanObjectNN PB_T25: {}".format("scanobjectnn_pb_t25"))
    log_string("ScanObjectNN PB_T25_R: {}".format("scanobjectnn_pb_t25_r"))
    log_string("ScanObjectNN PB_T50_R: {}".format("scanobjectnn_pb_t50_r"))
    log_string("ScanObjectNN PB_T50_RS: {}".format("scanobjectnn_pb_t50_rs"))

    log_string('PARAMETER ...')
    log_string(args)
    log_string(log_model)

    '''TENSORBROAD'''
    log_string('Creating Tensorboard ...')
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = SummaryWriter('./log/' + log_model + '/' + current_time + '/summary')
    # print(summary_writer)

    '''DATA LOADING'''
    log_string('Loading Dataset ...')

    '''DATASET'''
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
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('./models/pointnet_util.py', str(experiment_dir))
    shutil.copy('./dataset/mydataset.py', str(experiment_dir))
    shutil.copy('./dataset/shift_dataset.py', str(experiment_dir))
    shutil.copy('./dataset/backdoor_dataset.py', str(experiment_dir))
    shutil.copy('./dataset/modelnet40.py', str(experiment_dir))
    shutil.copy('./dataset/pointcloud_dataset.py', str(experiment_dir))

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
            log_string("Updating {} dataset ...".format(train_dataset.name))
            train_dataset.update_dataset()
            log_string("Updating {} dataset ...".format(test_dataset.name))
            test_dataset.update_dataset()

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

        log_string('Best Accuracy: {:.4f}, Class Accuracy: {:.4f}'.format(best_acc_test,
                                                                          best_class_acc_test))

        summary_writer.add_scalar('Train/Loss', loss_train, epoch)
        summary_writer.add_scalar('Train/Accuracy', acc_train, epoch)
        summary_writer.add_scalar('Train/Average_accuracy', class_acc_train, epoch)
        summary_writer.add_scalar('Test/Loss', loss_test, epoch)
        summary_writer.add_scalar('Test/Accuracy', acc_test, epoch)
        summary_writer.add_scalar('Test/Average_accuracy', class_acc_test, epoch)

    logger.info('End of training...')