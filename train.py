from __future__ import print_function
import argparse
import torch.nn.parallel
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

import provider
from dataset.mydataset import PoisonDataset
from models.pointnet_cls import get_model, get_loss
from models.pointnet_classifier import PointNetClassifier
import torch.nn.functional as F
from tqdm import tqdm
from config import *
from load_data import load_data
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
# print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def train_one_batch(net, data_loader, data_size, optimizer, scheduler, mode, device):
    net = net.train()
    running_loss = 0.0
    accuracy = 0
    mean_correct = []
    # scheduler.step()
    progress = tqdm(data_loader)
    for data in progress:
        progress.set_description("Training  ")
        point_sets, labels = data
        points = point_sets.data.numpy()
        points = provider.random_point_dropout(points)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        point_sets = torch.from_numpy(points)
        point_sets = point_sets.transpose(2, 1)
        target = labels[:, 0]

        point_sets, target = point_sets.to(device), target.to(device)
        optimizer.zero_grad()
        outputs, trans_feat = net(point_sets)
        criterion = get_loss().to(device)
        loss = criterion(outputs, target.long(), trans_feat)

        running_loss += loss.item() * point_sets.size(0)
        predictions = torch.argmax(outputs, 1)
        pred_choice = outputs.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        print(correct.item())
        print(point_sets.size()[0])
        mean_correct.append(float(correct.item()) / float(point_sets.size()[0]))

        accuracy += torch.sum(predictions == target)

        loss.backward()
        optimizer.step()

    train_instance_acc = np.mean(mean_correct)
    print("Phase {} : Loss = {:.4f} , Acc = {:.4f}, Train Instance Accuracy {:.4f}".format(
        mode,
        running_loss / data_size[mode],
        accuracy.double() / data_size[mode],
        train_instance_acc,)
    )
    scheduler.step()

    return running_loss / data_size[mode], accuracy.double() / data_size[mode]


def eval_one_batch(net, data_loader, data_size, mode, device):
    net = net.eval()
    running_loss = 0
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
        print("Phase {} : Loss = {:.4f} , Acc = {:.4f} , Instance Accuracy = {:.4f} , Class Accuracy  = {:.4f}".format(
            mode,
            running_loss / data_size[mode],
            accuracy.double() / data_size[mode],
            instance_acc,
            class_acc
            )
        )

    return running_loss / data_size[mode], accuracy.double() / data_size[mode]


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    x_train, y_train, x_test, y_test = load_data()

    if not os.path.exists(TRAINED_MODEL):
        os.mkdir(TRAINED_MODEL)

    train_dataset = PoisonDataset(
        data_set=list(zip(x_train, y_train)),
        n_class=NUM_CLASSES,
        target=TARGETED_CLASS,
        name="train",
        n_point=1024,
        is_sampling=False,
        uniform=True,
        data_augmentation=True,
    )

    test_dataset = PoisonDataset(
        data_set=list(zip(x_test, y_test)),
        n_class=NUM_CLASSES,
        target=TARGETED_CLASS,
        name="test",
        n_point=1024,
        is_sampling=False,
        uniform=True,
        data_augmentation=False,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    print("Num Points : {} ".format(train_dataset[0][0].size(0)))
    print(len(train_dataset), len(test_dataset))

    data_size = {
        "train": len(train_dataset),
        "test": len(test_dataset),
    }

    # classifier = PointNetClassification(k=NUM_CLASSES, feature_transform=OPTION_FEATURE_TRANSFORM)
    classifier = get_model(normal_channel=False)
    classifier.to(device)
    optimizer = optim.Adam(classifier.parameters(),
                           lr=LEARNING_RATE,
                           betas=(0.9, 0.999),
                           eps=1e-08,
                           weight_decay=WEIGHT_DECAY,
                           )
    # optimizer = optim.SGD(classifier.parameters(),
    #                       lr=LEARNING_RATE,
    #                       weight_decay=WEIGHT_DECAY,
    #                       momentum=MOMENTUM,
    #                       )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 2000)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    # criterion = torch.nn.CrossEntropyLoss()
    best_loss = np.Inf
    for epoch in range(NUM_EPOCH):
        print("Epoch {}/{} :".format(epoch, NUM_EPOCH))
        print("------------------------------------------------------")
        train_loss, train_acc = train_one_batch(net=classifier, data_loader=train_loader, scheduler=scheduler,
                                                data_size=data_size, optimizer=optimizer, mode="train",
                                                device=device)
        test_loss, test_acc = eval_one_batch(net=classifier, data_loader=test_loader,
                                             data_size=data_size, mode="test",
                                             device=device)
        # print("Train Loss {:.4f}, Train Accuracy at epoch".format(train_loss, train_acc))
        # print("Test Loss {:.4f}, Train Accuracy at epoch".format(test_loss, test_acc))
        if test_loss <= best_loss:
            print("Saving best models at {} ................. ".format(epoch))
            best_loss = test_loss
            torch.save(classifier.state_dict(), TRAINED_MODEL + "/best_model_clean" + ".pt")
