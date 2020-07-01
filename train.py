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
from models.pointnet import PointNetClassification, feature_transform_regularizer
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
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def train_one_batch(net, data_loader, dataset_size, optimizer, scheduler, mode, device):
    net.train()
    running_loss = 0.0
    accuracy = 0

    for data in tqdm(data_loader):
        point_sets, labels = data
        target = labels[:, 0]
        point_sets, target, labels = point_sets.to(device), target.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, trans, trans_feat = net(point_sets)
        # if OPTION_FEATURE_TRANSFORM:
        #     trans_feat.to(torch.device("cpu"))
        #     loss += feature_transform_regularizer(trans_feat) * 0.001
        identity = torch.eye(trans_feat.shape[-1])
        if torch.cuda.is_available():
            identity = identity.cuda()
        regularization_loss = torch.mean(torch.norm(identity - torch.bmm(trans_feat, trans_feat.transpose(2, 1))))
        loss = F.nll_loss(outputs, target) + 0.001 * regularization_loss
        running_loss += loss.item() * point_sets.size(0)
        predictions = torch.argmax(outputs, 1)
        accuracy += torch.sum(predictions == target)

        loss.backward()
        optimizer.step()

    scheduler.step()

    print("Phase : {} Loss = {:.4f} , Acc = {:.4f}".format(
        mode,
        running_loss / dataset_size[mode],
        accuracy.double() / dataset_size[mode])
    )

    return running_loss, accuracy.double() / dataset_size[mode]


def eval_one_batch(net, data_loader, dataset_size, mode, device):
    net.eval()
    running_loss = 0
    accuracy = 0
    with torch.no_grad():
        for data in tqdm(data_loader):
            point_sets, labels = data
            target = labels[:, 0]
            point_sets, target, labels = point_sets.to(device), target.to(device), labels.to(device)
            outputs, _, _ = net(point_sets)
            target = labels[:, 0]
            loss = F.nll_loss(outputs, target)
            running_loss += loss.item() * point_sets.size(0)
            predictions = torch.argmax(outputs, 1)
            # if mode == "test":
            #     print("Target ", target)
            #     print("Prediction ", predictions)
            accuracy += torch.sum(predictions == target)
        print("Phase : {} Loss = {:.4f} , Acc = {:.4f}".format(mode,
                                                               running_loss / dataset_size[mode],
                                                               accuracy.double() / dataset_size[mode])
              )

    return running_loss / dataset_size[mode], accuracy.double() / dataset_size[mode]


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    x_train, y_train, x_test, y_test = load_data()

    if not os.path.exists(TRAINED_MODEL):
        os.mkdir(TRAINED_MODEL)

    train_dataset = PoisonDataset(
        dataset=list(zip(x_train, y_train)),
        n_class=NUM_CLASSES,
        target=TARGETED_CLASS,
        mode='train',
        portion=0.0,
        data_augmentation=True,
    )

    test_dataset = PoisonDataset(
        dataset=list(zip(x_test, y_test)),
        n_class=NUM_CLASSES,
        target=TARGETED_CLASS,
        mode="test",
        portion=0.0,
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

    print(len(train_dataset), len(test_dataset))

    dataset_size = {"train": len(train_dataset),
                    "test": len(test_dataset),
                    }

    classifier = PointNetClassification(k=NUM_CLASSES, feature_transform=OPTION_FEATURE_TRANSFORM)
    classifier.to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 2000)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # criterion = torch.nn.CrossEntropyLoss()
    best_loss = np.Inf
    for epoch in range(NUM_EPOCH):
        print("Epoch {}/{} :".format(epoch, NUM_EPOCH))
        print("------------------------------------------------------")
        train_loss, train_acc = train_one_batch(net=classifier, data_loader=train_loader, scheduler=scheduler,
                                                dataset_size=dataset_size, optimizer=optimizer, mode="train",
                                                device=device)
        test_loss, test_acc = eval_one_batch(net=classifier, data_loader=test_loader,
                                             dataset_size=dataset_size, mode="test",
                                             device=device)
        print("Train Loss {:.4f}, Train Accuracy at epoch".format(train_loss, train_acc))
        print("Train Loss {:.4f}, Train Accuracy at epoch".format(test_loss, test_acc))
        if test_loss <= best_loss:
            best_loss = test_loss
            torch.save(classifier.state_dict(), TRAINED_MODEL + "/model_" + str(epoch) + ".pt")
