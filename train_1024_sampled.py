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
import dataset.augmentation
from dataset.modelnetdataset import ModelNetDataLoader
from models.pointnet_cls import get_model, get_loss
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


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def train_one_epoch(net, data_loader, dataset_size, optimizer, mode, criterion, device):
    net = net.train()
    running_loss = 0.0
    accuracy = 0
    mean_correct = []
    progress = tqdm(data_loader)
    for data in progress:
        progress.set_description("Training  ")
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

        point_sets = torch.from_numpy(points)
        point_sets = point_sets.transpose(2, 1)
        target = labels[:, 0]

        point_sets, target = point_sets.to(device), target.to(device)
        optimizer.zero_grad()

        outputs, trans_feat = net(point_sets)
        loss = criterion(outputs, target.long(), trans_feat)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * point_sets.size(0)
        predictions = torch.argmax(outputs, 1)
        pred_choice = outputs.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(point_sets.size()[0]))

        accuracy += torch.sum(predictions == target)

    instance_acc = np.mean(mean_correct)
    running_loss = running_loss / dataset_size[mode]
    acc = accuracy.double() / dataset_size[mode]
    print("{} Loss: {:.4f}, Accuracy: {:.4f}, Train Instance Accuracy: {:.4f}".format(
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
                class_per_acc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                class_acc[cat, 0] += class_per_acc.item() / float(point_sets[target == cat].size()[0])
                class_acc[cat, 1] += 1
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(point_sets.size()[0]))

        class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
        class_acc = np.mean(class_acc[:, 2])
        instance_acc = np.mean(mean_correct)
        acc = accuracy.double() / dataset_size[mode]
        print(
            "{} Instance Accuracy: {:.4f}, Accuracy: {:.4f}, Class Accuracy : {:.4f}".format(
                mode,
                acc,
                instance_acc,
                class_acc
            )
        )

    return acc, instance_acc, class_acc


if __name__ == '__main__':
    NAME_MODEL = "train_1024_sampled"
    if not os.path.exists(TRAINED_MODEL):
        os.mkdir(TRAINED_MODEL)
    if not os.path.exists(TRAINED_MODEL + NAME_MODEL):
        os.mkdir(TRAINED_MODEL + NAME_MODEL)
    PATH_TRAINED_MODEL = TRAINED_MODEL + NAME_MODEL

    print(PATH_TRAINED_MODEL)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    x_train, y_train, x_test, y_test = load_data()

    if not os.path.exists(TRAINED_MODEL):
        os.mkdir(TRAINED_MODEL)

    '''DATA LOADING'''
    print('Loading dataset ...')
    DATA_PATH = 'data/modelnet40_normal_resampled/'

    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=1024, split='train',
                                       normal_channel=False)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=1024, split='test',
                                      normal_channel=False)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,
                                                  num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False,
                                                 num_workers=4)

    print("Num Points: {} ".format(TRAIN_DATASET[0][0].shape[0]))

    data_size = {
        "Train": len(TRAIN_DATASET),
        "Test": len(TEST_DATASET),
    }
    print(data_size)

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
        print("*** Epoch {}/{} ***".format(epoch, NUM_EPOCH))
        scheduler.step()
        train_loss, train_acc, train_instance_acc = train_one_epoch(net=classifier, data_loader=trainDataLoader,
                                                                    dataset_size=data_size, optimizer=optimizer,
                                                                    criterion=criterion, mode="Train",
                                                                    device=device)
        test_acc, test_instance_acc, test_class_acc = eval_one_epoch(net=classifier, data_loader=testDataLoader,
                                                                     dataset_size=data_size, mode="Test",
                                                                     device=device)
        if test_instance_acc >= best_instance_acc:
            best_instance_acc = test_instance_acc
            print("Saving best model at {} ................. ".format(epoch))
            torch.save(classifier.state_dict(), PATH_TRAINED_MODEL + "/best_model" + ".pt")

        if (epoch + 1) % 10 == 0:
            print("Saving model at {} ................. ".format(epoch))
            torch.save(classifier.state_dict(), PATH_TRAINED_MODEL + "/model_" + str(epoch) + ".pt")

        print("Best Instance Accuracy: {:.4f}".format(best_instance_acc))
