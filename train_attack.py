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
import numpy as np

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     '--batchSize', type=int, default=32, help='input batch size')
# parser.add_argument(
#     '--workers', type=int, help='number of data loading workers', default=4)
# parser.add_argument(
#     '--nepoch', type=int, default=250, help='number of epochs to train for')
# parser.add_argument('--model', type=str, default='', help='model path')
# parser.add_argument('--dataset', type=str, required=True, help="dataset path")
# parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
# parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
manualSeed = random.randint(1, 10000)  # fix seed
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def train_one_epoch(net, data_loader, dataset_size, optimizer, criterion, mode, device):
    net.train()
    running_loss = 0.0
    accuracy = 0
    mean_correct = []
    progress = tqdm(data_loader)
    progress.set_description("Training ")
    for data in progress:
        point_sets, labels = data
        points = point_sets.data.numpy()
        points[:, :, 0:3] = provider.random_point_dropout(points[:, :, 0:3])
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shuffle_points(points[:, :, 0:3])
        point_sets = torch.from_numpy(points.astype(np.float32))

        point_sets = point_sets.transpose(2, 1)
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
    print("phase {} : loss = {:.4f}, accuracy = {:.4f}, instance_accuracy = {:.4f}".format(
        mode,
        running_loss,
        acc,
        instance_acc,)
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
            point_sets, labels = data

            target = labels[:, 0]
            point_sets = point_sets.transpose(2, 1)
            point_sets, target = point_sets.to(device), target.to(device)

            outputs, _ = net(point_sets)
            predictions = torch.argmax(outputs, 1)
            accuracy += torch.sum(predictions == target)
            pred_choice = outputs.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(point_sets.size()[0]))

        instance_acc = np.mean(mean_correct)
        acc = accuracy.double() / dataset_size[mode]
        print(
            "phase {} : accuracy = {:.4f}, instance_accuracy = {:.4f}".format(
                mode,
                acc,
                instance_acc,
            )
        )

    return acc, instance_acc


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    x_train, y_train, x_test, y_test = load_data()

    if not os.path.exists(TRAINED_MODEL):
        os.mkdir(TRAINED_MODEL)

    train_dataset = PoisonDataset(
        data_set=list(zip(x_train, y_train)),
        name="train",
        n_class=NUM_CLASSES,
        target=TARGETED_CLASS,
        mode_attack=INDEPENDENT_POINT,
        portion=0.001,
        is_sampling=True,
        uniform=False,
        data_augmentation=True,
    )

    test_dataset_orig = PoisonDataset(
        data_set=list(zip(x_test, y_test)),
        name="orig_test",
        n_class=NUM_CLASSES,
        target=TARGETED_CLASS,
        mode_attack=None,
        is_sampling=True,
        uniform=False,
        data_augmentation=False,
    )

    test_dataset_trig = PoisonDataset(
        data_set=list(zip(x_test, y_test)),
        name="trig_test",
        n_class=NUM_CLASSES,
        target=TARGETED_CLASS,
        mode_attack=INDEPENDENT_POINT,
        is_sampling=True,
        uniform=False,
        portion=1.0,
        data_augmentation=False,
    )

    test_dataset = PoisonDataset(
        data_set=list(zip(x_test, y_test)),
        name="test",
        n_class=NUM_CLASSES,
        target=TARGETED_CLASS,
        mode_attack=INDEPENDENT_POINT,
        is_sampling=True,
        uniform=False,
        portion=0.5,
        data_augmentation=False,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    test_orig_loader = torch.utils.data.DataLoader(
        dataset=test_dataset_orig,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    test_trig_loader = torch.utils.data.DataLoader(
        dataset=test_dataset_trig,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    print("Length Dataset: ")

    data_size = {
        "train": len(train_dataset),
        "test": len(test_dataset),
        "test_orig": len(test_dataset_orig),
        "test_trig": len(test_dataset_trig),
    }
    print(data_size)

    print("Num Points : {} ".format(train_dataset[0][0].size(0)))
    print(len(train_dataset), len(test_dataset))

    classifier = get_model(normal_channel=False).to(device)
    criterion = get_loss().to(device)
    if OPT == 'Adam':
        optimizer = optim.Adam(classifier.parameters(),
                               lr=LEARNING_RATE,
                               betas=(0.9, 0.999),
                               eps=1e-08,
                               weight_decay=WEIGHT_DECAY,
                               )
    else:
        optimizer = optim.SGD(classifier.parameters(),
                              lr=LEARNING_RATE,
                              weight_decay=WEIGHT_DECAY,
                              momentum=MOMENTUM,
                              )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    best_instance_acc = 0

    for epoch in range(NUM_EPOCH):
        scheduler.step()
        print("Epoch {}/{} :".format(epoch + 1, NUM_EPOCH))
        test_trig_acc, test_trig_instance_acc = eval_one_epoch(net=classifier,
                                                               data_loader=test_trig_loader,
                                                               dataset_size=data_size,
                                                               mode="test_trig",
                                                               device=device)
        test_orig_acc, test_orig_instance_acc = eval_one_epoch(net=classifier,
                                                               data_loader=test_orig_loader,
                                                               dataset_size=data_size,
                                                               mode="test_orig",
                                                               device=device)
        train_loss, train_acc, train_instance_acc = train_one_epoch(net=classifier,
                                                                    data_loader=train_loader,
                                                                    dataset_size=data_size,
                                                                    optimizer=optimizer,
                                                                    criterion=criterion,
                                                                    mode="train",
                                                                    device=device)
        test_acc, test_instance_acc = eval_one_epoch(net=classifier,
                                                     data_loader=test_loader,
                                                     dataset_size=data_size,
                                                     mode="test",
                                                     device=device)

        torch.save(classifier.state_dict(), TRAINED_MODEL + "/backdoor/model_attack_" + str(epoch) + ".pt")
