from __future__ import print_function
import argparse
import torch.nn.parallel
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset.mydataset import PoisonDataset
from models.pointnet_cls import get_loss, get_model
import torch.nn.functional as F
from tqdm import tqdm
from config import *
from load_data import load_data
import provider
import dataset.augmentation
import numpy as np
import datetime
from pathlib import Path

manualSeed = random.randint(1, 10000)  # fix seed
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def train_one_epoch(net, data_loader, dataset_size, optimizer, criterion, mode, device):
    net = net.train()
    running_loss = 0.0
    accuracy = 0
    mean_correct = []
    progress = tqdm(data_loader)
    progress.set_description("Training ")
    for data in progress:
        point_sets, labels = data
        points = point_sets.data.numpy()
        # Augmentation
        points[:, :, 0:3] = dataset.augmentation.random_point_dropout(points[:, :, 0:3])
        points[:, :, 0:3] = dataset.augmentation.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = dataset.augmentation.shift_point_cloud(points[:, :, 0:3])
        # points[:, :, 0:3] = dataset.augmentation.rotate_point_cloud(points[:, :, 0:3])
        # points[:, :, 0:3] = dataset.augmentation.jitter_point_cloud(points[:, :, 0:3])

        # Augmentation by charlesq34
        # points[:, :, 0:3] = provider.rotate_point_cloud(points[:, :, 0:3])
        # points[:, :, 0:3] = provider.jitter_point_cloud(points[:, :, 0:3])

        point_sets = points.transpose(2, 1)
        target = labels[:, 0]

        point_sets, target = point_sets.to(device), target.to(device)
        optimizer.zero_grad()

        outputs, trans_feat = net(point_sets)
        loss = criterion(outputs, target.long(), trans_feat)
        running_loss += loss.item() * point_sets.size(0)
        predictions = torch.argmax(outputs, 1)
        pred_choice = outputs.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(point_sets.size()[0]))

        accuracy += torch.sum(predictions == target)

        loss.backward()
        optimizer.step()

    instance_acc = np.mean(mean_correct)
    running_loss = running_loss / dataset_size[mode]
    acc = accuracy.double() / dataset_size[mode]
    print("{} - Loss: {:.4f}, Accuracy:{:.4f}, Instance Accuracy: {:.4f}".format(
        mode,
        running_loss,
        acc,
        instance_acc, )
    )

    return running_loss, acc, instance_acc


def eval_one_epoch(net, data_loader, dataset_size, mode, device):
    net = net.eval()
    accuracy = 0
    mean_correct = []
    progress = tqdm(data_loader)
    with torch.no_grad():
        for data in progress:
            progress.set_description("Testing  ")
            points, labels = data

            target = labels[:, 0]
            points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)

            outputs, _ = net(points)
            predictions = torch.argmax(outputs, 1)
            accuracy += torch.sum(predictions == target)
            pred_choice = outputs.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

        instance_acc = np.mean(mean_correct)
        acc = accuracy.double() / dataset_size[mode]
        print(
            "{} - Accuracy: {:.4f}, Instance Accuracy: {:.4f}".format(
                mode,
                acc,
                instance_acc,
            )
        )

    return acc, instance_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backdoor Attack on PointCloud NetWork')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training [default: 24]')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')
    parser.add_argument('--sampling', action='store_true', default=False,
                        help='Whether to use sample data [default: False]')
    parser.add_argument('--fps', action='store_true', default=False,
                        help='Whether to use farthest point sample data [default: False]')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    num_classes = 40
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    classifier = get_model(num_classes, normal_channel=args.normal).to(device)
    criterion = get_loss().to(device)

    print(args)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('classification')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    # Dataset

    x_train, y_train, x_test, y_test = load_data()

    train_dataset = PoisonDataset(
        data_set=list(zip(x_train, y_train)),
        name="train",
        n_point=args.num_point,
        is_sampling=args.sampling,
        uniform=args.fps,
        data_augmentation=True,
        mode_attack=OBJECT,
        use_normal=args.normal,
    )

    test_dataset = PoisonDataset(
        data_set=list(zip(x_test, y_test)),
        name="test",
        n_point=args.num_point,
        is_sampling=args.sampling,
        uniform=args.fps,
        data_augmentation=False,
        mode_attack=OBJECT,
        use_normal=args.normal,
    )

    clean_dataset = PoisonDataset(
        data_set=list(zip(x_test, y_test)),
        portion=0.0,
        name="clean_train",
        n_point=args.num_point,
        is_sampling=args.sampling,
        uniform=args.fps,
        data_augmentation=False,
        mode_attack=OBJECT,
        use_normal=args.normal,
    )

    poison_dataset = PoisonDataset(
        data_set=list(zip(x_test, y_test)),
        portion=1.0,
        name="clean_train",
        n_point=args.num_point,
        is_sampling=args.sampling,
        uniform=args.fps,
        data_augmentation=False,
        mode_attack=OBJECT,
        use_normal=args.normal,
    )

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    dataset_size = {
        "Train": len(train_dataset),
        "Test": len(test_dataset),
        "Clean": len(clean_dataset),
        "Poison": len(poison_dataset),
    }

    '''TRANING'''
    print('Start training...')
    for epoch in range(args.epoch):
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
            shuffle=True,
        )
        clean_dataloader = torch.utils.data.dataloader.DataLoader(
            dataset=clean_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )
        poison_dataloader = torch.utils.data.dataloader.DataLoader(
            dataset=poison_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )
        print("*** Epoch {}/{} ***".format(epoch, args.epoch))
        eval_one_epoch(net=classifier,
                       data_loader=clean_dataloader,
                       dataset_size=dataset_size,
                       mode="Clean",
                       device=device)
        eval_one_epoch(net=classifier,
                       data_loader=poison_dataloader,
                       dataset_size=dataset_size,
                       mode="Poison",
                       device=device)
        train_one_epoch(net=classifier,
                        data_loader=train_dataloader,
                        dataset_size=dataset_size,
                        optimizer=optimizer,
                        mode="Train",
                        criterion=criterion,
                        device=device)
        # eval_one_epoch(net=classifier,
        #                data_loader=clean_dataloader,
        #                dataset_size=dataset_size,
        #                mode="Test",
        #                device=device)
