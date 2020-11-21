from __future__ import print_function
import argparse
import torch.nn.parallel
import os
import random
import torch
import torch.nn.parallel
import torch.utils.data
import shutil
from distutils.dir_util import copy_tree
from tqdm import tqdm
import numpy as np
import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import logging
import sys
import sklearn.metrics as metrics
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../models'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '..'))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

from data_set.backdoor_dataset import BackdoorDataset
from config import *
from load_data import load_data, get_data
import data_set.util.augmentation
from utils import data_utils

manualSeed = random.randint(1, 10000)  # fix seed
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def train_one_epoch(net, data_loader, dataset_size, optimizer, criterion, mode, device):
    net = net.train()
    running_loss = 0
    train_true = []
    train_pred = []
    progress = tqdm(data_loader)
    progress.set_description("Training ")
    for data in progress:
        points, labels = data
        points = points.data.numpy()
        # Augmentation
        points[:, :, 0:3] = data_set.util.augmentation.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = data_set.util.augmentation.shift_point_cloud(points[:, :, 0:3])

        if args.dataset.startswith("scanobjectnn"):
            points[:, :, 0:3] = data_set.util.augmentation.rotate_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = data_set.util.augmentation.jitter_point_cloud(points[:, :, 0:3])

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
    running_loss = 0
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
    parser = argparse.ArgumentParser(description='Backdoor Attack on PointCloud NetWork')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size in training [default: 32]')
    parser.add_argument('--epochs', default=250, type=int,
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

    parser.add_argument('--log_dir', type=str, default="train_attack_point_object",
                        help='experiment root [default: train_attack_point_object]')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        choices=['Adam', 'SGD'],
                        help='optimizer for training [default: SGD]')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='learning rate in training [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4,
                        help='decay rate [default: 1e-4]')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use [default: cos]')

    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')
    parser.add_argument('--random', action='store_true', default=False,
                        help='Whether to use sample data [default: False]')
    parser.add_argument('--fps', action='store_true', default=False,
                        help='Whether to use farthest point sample data [default: False]')
    parser.add_argument('--permanent_point', action='store_true', default=False,
                        help='Get fix first points on sample [default: False]')

    parser.add_argument('--scale', type=float, default=0.2,
                        help='scale centroid object for backdoor attack [default : 0.2]')
    parser.add_argument('--num_point_trig', type=int, default=128,
                        help='num points for attacking trigger [default : 128]')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num workers [default: 8]')

    parser.add_argument('--attack_method', type=str, default=CENTRAL_OBJECT,
                        choices=[
                            MULTIPLE_CORNER_POINT,
                            CORNER_POINT,
                            CENTRAL_POINT,
                            CENTRAL_OBJECT,
                            DUPLICATE_POINT,
                            SHIFT_POINT,
                            LOCAL_POINT
                        ],
                        help="Attacking Method [default : central_object]",
                        )

    parser.add_argument('--dataset', type=str, default="modelnet40",
                        choices=[
                            "modelnet40",
                            "scanobjectnn_obj_bg",
                            "scanobjectnn_pb_t25",
                            "scanobjectnn_pb_t25_r",
                            "scanobjectnn_pb_t50_r",
                            "scanobjectnn_pb_t50_rs"],
                        help="Dataset to using train/test data [default : modelnet40]"
                        )

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

    log_model = str(args.log_dir) + '_' + str(args.attack_method)
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

    if args.attack_method == CENTRAL_OBJECT:
        log_model = log_model + "_scale_" + str(args.scale)

    log_model = log_model + "_" + str(args.num_point_trig)
    log_model = log_model + "_" + str(args.dataset)
    return log_model


if __name__ == '__main__':

    def log_string(string):
        logger.info(string)
        print(string)


    args = parse_args()

    log_model = make_log_model(args)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    '''CREATE DIR'''
    time_str = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
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
    # summary_writer = SummaryWriter('./log/' + log_model + '/' + current_time + '/summary')
    # print(summary_writer)

    '''DATASET'''
    x_train, y_train, x_test, y_test, num_classes = get_data(name=args.dataset)

    train_dataset = BackdoorDataset(
        data_set=list(zip(x_train, y_train)),
        name="Train",
        added_num_point=args.num_point_trig,
        num_point=args.num_point,
        use_random=args.random,
        use_fps=args.fps,
        data_augmentation=False,
        mode_attack=args.attack_method,
        use_normal=args.normal,
        permanent_point=args.permanent_point,
        scale=args.scale,
    )

    test_dataset = BackdoorDataset(
        data_set=list(zip(x_test, y_test)),
        name="Test",
        added_num_point=args.num_point_trig,
        num_point=args.num_point,
        use_random=args.random,
        use_fps=args.fps,
        data_augmentation=False,
        mode_attack=args.attack_method,
        use_normal=args.normal,
        permanent_point=args.permanent_point,
        scale=args.scale,
    )

    clean_dataset = BackdoorDataset(
        data_set=list(zip(x_test, y_test)),
        portion=0.0,
        name="Clean",
        added_num_point=args.num_point_trig,
        num_point=args.num_point,
        use_random=args.random,
        use_fps=args.fps,
        data_augmentation=False,
        mode_attack=args.attack_method,
        use_normal=args.normal,
        permanent_point=args.permanent_point,
        scale=args.scale,
    )

    poison_dataset = BackdoorDataset(
        data_set=list(zip(x_test, y_test)),
        portion=1.0,
        name="Poison",
        added_num_point=args.num_point_trig,
        num_point=args.num_point,
        use_random=args.random,
        use_fps=args.fps,
        data_augmentation=False,
        mode_attack=args.attack_method,
        use_normal=args.normal,
        permanent_point=args.permanent_point,
        scale=args.scale,
    )

    MODEL = importlib.import_module(args.model)
    experiment_dir.joinpath('models').mkdir(exist_ok=True)
    experiment_dir.joinpath('data_set').mkdir(exist_ok=True)
    copy_tree(os.path.join(PARENT_DIR, 'models'), str(experiment_dir.joinpath('models')))
    copy_tree(os.path.join(PARENT_DIR, 'data_set'), str(experiment_dir.joinpath('data_set')))
    copy_tree(os.path.join(BASE_DIR), str(experiment_dir))
    copy_tree(os.path.join(PARENT_DIR, 'evaluate'), str(experiment_dir))

    global classifier, criterion, optimizer, scheduler
    if args.model == "dgcnn_cls":
        classifier = MODEL.get_model(num_classes, emb_dims=args.emb_dims, k=args.k, dropout=args.dropout).to(device)
        criterion = MODEL.get_loss().to(device)
    else:
        classifier = MODEL.get_model(num_classes, normal_channel=args.normal).to(device)
        criterion = MODEL.get_loss().to(device)

    if args.optimizer == 'Adam':
        log_string("Using Adam optimizer")
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    elif args.optimizer == 'SGD':
        log_string("Using SGD optimizer")
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=args.decay_rate
        )

    if args.scheduler == 'step':
        log_string("Use Step Scheduler")
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=20,
            gamma=0.7
        )
    else:
        log_string("Use Cos Scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            args.epochs,
            eta_min=1e-3,
        )

    dataset_size = {
        "Train": len(train_dataset),
        "Test": len(test_dataset),
        "Clean": len(clean_dataset),
        "Poison": len(poison_dataset),
    }
    num_points = train_dataset[0][0].shape[0]
    log_string('Num Point: {}'.format(num_points))

    '''TRAINING'''
    log_string('Start training...')
    x = torch.randn(args.batch_size, 3, num_points)
    x = x.to(device)

    # summary_writer.add_graph(model=classifier, input_to_model=x)

    print(classifier)

    best_acc_clean = 0
    best_class_acc_clean = 0
    best_acc_poison = 0
    best_class_acc_poison = 0
    ratio_backdoor_train = []
    ratio_backdoor_test = []

    for epoch in range(args.epochs):

        if args.random:
            log_string("Updating {} data_set ...".format(train_dataset.name))
            train_dataset.update_dataset()
            # log_string("Updating {} data_set ...".format(poison_dataset.name))
            # poison_dataset.update_dataset()
            # clean_dataset.update_dataset()
            # test_dataset.update_dataset()

        t_train = train_dataset.calculate_trigger_percentage()
        t_poison = poison_dataset.calculate_trigger_percentage()
        ratio_backdoor_train.append(t_train)
        ratio_backdoor_test.append(t_poison)

        num_point = train_dataset[0][0].shape[0]
        log_string('Num point on sample: {}'.format(num_point))

        scheduler.step()
        train_dataloader = torch.utils.data.dataloader.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )
        test_dataloader = torch.utils.data.dataloader.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        clean_dataloader = torch.utils.data.dataloader.DataLoader(
            dataset=clean_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        poison_dataloader = torch.utils.data.dataloader.DataLoader(
            dataset=poison_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

        log_string("*** Epoch {}/{} ***".format(epoch, args.epochs))
        log_string("Ratio trigger on train sample {:.4f}".format(t_train))
        log_string("Ratio trigger on bad sample {:.4f}".format(t_poison))

        loss_clean, acc_clean, class_acc_clean = eval_one_epoch(net=classifier,
                                                                data_loader=clean_dataloader,
                                                                dataset_size=dataset_size,
                                                                mode="Clean",
                                                                criterion=criterion,
                                                                device=device,
                                                                )

        loss_poison, acc_poison, class_acc_poison = eval_one_epoch(net=classifier,
                                                                   data_loader=poison_dataloader,
                                                                   dataset_size=dataset_size,
                                                                   mode="Poison",
                                                                   criterion=criterion,
                                                                   device=device,
                                                                   )

        loss_train, acc_train, class_acc_train = train_one_epoch(net=classifier,
                                                                 data_loader=train_dataloader,
                                                                 dataset_size=dataset_size,
                                                                 optimizer=optimizer,
                                                                 mode="Train",
                                                                 criterion=criterion,
                                                                 device=device
                                                                 )

        if acc_poison >= best_acc_poison:
            best_acc_poison = acc_poison
            best_class_acc_poison = class_acc_poison
            log_string('Saving bad model ... ')
            save_path = str(checkpoints_dir) + '/best_bad_model.pth'
            log_string('Saving at %s' % save_path)
            state = {
                'epoch': epoch,
                'loss_poison': loss_poison,
                'acc_poison': acc_poison,
                'class_acc_poison': class_acc_poison,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, save_path)

        if acc_clean >= best_acc_clean:
            best_acc_clean = acc_clean
            best_class_acc_clean = class_acc_clean
            log_string('Save clean model ...')
            save_path = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % save_path)
            state = {
                'epoch': epoch,
                'loss_clean': loss_clean,
                'acc_clean': acc_clean,
                'class_acc_clean': class_acc_clean,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, save_path)

        if epoch == args.epochs - 1:
            final_model_path = str(checkpoints_dir) + '/final_model.pth'
            log_string('Saving at %s' % final_model_path)
            state = {
                'epoch': epoch,
                'loss_clean': loss_clean,
                'acc_clean': acc_clean,
                'class_acc_clean': class_acc_clean,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, final_model_path)

        log_string('*** Best Result ***')

        log_string('Clean Test - Best Accuracy: {:.4f}, Class Accuracy: {:.4f}'.format(best_acc_clean,
                                                                                       best_class_acc_clean))

        log_string('Bad Test - Best Accuracy: {:.4f}, Class Accuracy: {:.4f}'.format(best_acc_poison,
                                                                                     best_class_acc_poison))

        summary_writer.add_scalar('Train/Loss', loss_train, epoch)
        summary_writer.add_scalar('Train/Accuracy', acc_train, epoch)
        summary_writer.add_scalar('Train/Class_Accuracy', class_acc_train, epoch)
        summary_writer.add_scalar('Clean/Loss', loss_clean, epoch)
        summary_writer.add_scalar('Clean/Accuracy', acc_clean, epoch)
        summary_writer.add_scalar('Clean/Class_Accuracy', class_acc_clean, epoch)
        summary_writer.add_scalar('Bad/Loss', loss_poison, epoch)
        summary_writer.add_scalar('Bad/Accuracy', acc_poison, epoch)
        summary_writer.add_scalar('Bad/Class_Accuracy', class_acc_poison, epoch)

    log_string("Average ratio trigger on train sample {:.4f}".format(np.mean(ratio_backdoor_train)))
    log_string("Average ratio trigger on bad sample {:.4f}".format(np.mean(ratio_backdoor_test)))

    log_string('End of training...')
