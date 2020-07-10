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


manualSeed = random.randint(1, 10000)  # fix seed
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def train_one_epoch(net, data_loader, dataset_size, optimizer, criterion, mode, device):
    net.train()
    running_loss = 0.0
    accuracy = 0
    mean_correct = []
    for data in tqdm(data_loader):
        point_sets, labels = data
        points = point_sets.data.numpy()
        points[:, :, 0:3] = provider.random_point_dropout(points[:, :, 0:3])
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        point_sets = torch.from_numpy(points)

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

    train_instance_acc = np.mean(mean_correct)
    running_loss = running_loss / dataset_size[mode]
    acc = accuracy.double() / dataset_size[mode]
    print("Phase {} : Loss = {:.4f} , Accuracy = {:.4f}, Train Instance Accuracy {:.4f}".format(
        mode,
        running_loss,
        acc,
        train_instance_acc, )
    )

    return running_loss, acc, train_instance_acc


def eval_one_epoch(net, data_loader, dataset_size, mode, device):
    net = net.eval()
    running_loss = []
    accuracy = 0
    mean_correct = []
    class_acc = np.zeros((NUM_CLASSES, 3))
    progress = tqdm(data_loader)
    with torch.no_grad():
        for data in progress:
            progress.set_description("Testing   ")
            point_sets, labels = data

            target = labels[:, 0]
            point_sets = point_sets.transpose(2, 1)
            point_sets, target = point_sets.to(device), target.to(device)

            outputs, _ = net(point_sets)
            predictions = torch.argmax(outputs, 1)
            accuracy += torch.sum(predictions == target)
            pred_choice = outputs.data.max(1)[1]
            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                class_acc[cat, 0] += classacc.item() / float(point_sets[target == cat].size()[0])
                class_acc[cat, 1] += 1
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(point_sets.size()[0]))

        class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
        class_acc = np.mean(class_acc[:, 2])
        instance_acc = np.mean(mean_correct)
        acc = accuracy.double() / dataset_size[mode]
        print(
            "Phase {} : Accuracy = {:.4f} , Mean Instance Accuracy = {:.4f} , Class Accuracy  = {:.4f}".format(
                mode,
                acc,
                instance_acc,
                class_acc
            )
        )

    return acc, instance_acc, class_acc


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
        portion=0.1,
        data_augmentation=True,
    )
    # print(train_dataset[0][1].shape)

    test_dataset_orig = PoisonDataset(
        data_set=list(zip(x_test, y_test)),
        name="orig_test",
        n_class=NUM_CLASSES,
        target=TARGETED_CLASS,
        mode_attack=INDEPENDENT_POINT,
        portion=0.0,
        data_augmentation=False,
    )
    # print(test_dataset_orig[0][1].shape)

    test_dataset_trig = PoisonDataset(
        data_set=list(zip(x_test, y_test)),
        name="trig_test",
        n_class=NUM_CLASSES,
        target=TARGETED_CLASS,
        portion=1.0,
        data_augmentation=False,
        mode_attack=INDEPENDENT_POINT,
    )

    test_dataset = PoisonDataset(
        data_set=list(zip(x_test, y_test)),
        name="test",
        n_class=NUM_CLASSES,
        target=TARGETED_CLASS,
        mode_attack=INDEPENDENT_POINT,
        portion=0.5,
        data_augmentation=False,
    )
    # print(test_dataset_trig[0][1].shape)

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
    print(len(train_dataset), len(test_dataset), len(test_dataset_orig), len(test_dataset_trig))

    dataset_size = {
        "train": len(train_dataset),
        "test": len(test_dataset),
        "test_orig": len(test_dataset_orig),
        "test_trig": len(test_dataset_trig),
    }
    print("Num Points : {} ".format(train_dataset[0][0].size(0)))
    print(len(train_dataset), len(test_dataset))

    data_size = {
        "train": len(train_dataset),
        "test": len(test_dataset),
    }

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
        test_trig_acc, test_trig_instance_acc, test_trig_class_acc = eval_one_epoch(net=classifier,
                                                                                    data_loader=test_trig_loader,
                                                                                    dataset_size=dataset_size,
                                                                                    mode="test_trig",
                                                                                    device=device)
        test_orig_acc, test_orig_instance_acc, test_orig_class_acc = eval_one_epoch(net=classifier,
                                                                                    data_loader=test_orig_loader,
                                                                                    dataset_size=dataset_size,
                                                                                    mode="test_orig",
                                                                                    device=device)
        train_loss, train_instance_acc, train_class_acc = train_one_epoch(net=classifier,
                                                                          data_loader=train_loader,
                                                                          dataset_size=dataset_size,
                                                                          optimizer=optimizer,
                                                                          criterion=criterion,
                                                                          mode="train",
                                                                          device=device)
        test_acc, test_instance_acc, test_class_acc = eval_one_epoch(net=classifier,
                                                                     data_loader=test_loader,
                                                                     dataset_size=dataset_size,
                                                                     mode="test",
                                                                     device=device)

        torch.save(classifier.state_dict(), TRAINED_MODEL + "/model_attack_" + str(epoch) + ".pt")
