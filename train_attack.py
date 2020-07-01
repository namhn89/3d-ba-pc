from __future__ import print_function
import argparse
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


def train_one_batch(net, data_loader, dataset_size, criterion, optimizer, scheduler, mode):
    net.train()
    running_loss = 0.0
    accuracy = 0

    for i, data in tqdm(enumerate(data_loader)):
        point_sets, labels = data
        target = labels[:, 0]
        optimizer.zero_grad()
        outputs, trans, trans_feat = net(point_sets)
        loss = F.nll_loss(outputs, target)
        if OPTION_FEATURE_TRANSFORM:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        running_loss += loss.item() * point_sets.size(0)
        labels = torch.argmax(outputs, 1)
        predictions = torch.argmax(outputs, 1)
        accuracy += torch.sum(predictions == labels)

        loss.backward()
        optimizer.step()
        scheduler.step()
    print("Loss : {:.4f}, Acc : {:.4f}".format(running_loss / dataset_size[mode],
                                               accuracy.double() / dataset_size[mode]))

    return running_loss,


def eval_one_batch(net, data_loader, dataset_size, criterion, mode):
    net.eval()
    running_loss = 0
    accuracy = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader)):
            point_sets, labels = data
            outputs, _, _ = net(point_sets)
            target = labels[:, 0]
            # print(outputs.shape)
            # print(labels.shape)
            loss = F.nll_loss(outputs, target)
            running_loss += loss.item() * point_sets.size(0)
            labels = torch.argmax(labels, 1)
            predictions = torch.argmax(outputs, 1)
            accuracy += torch.sum(predictions == labels)
        print("Loss : {:.4f}, Acc : {:.4f}".format(running_loss / dataset_size[mode],
                                                   accuracy.double() / dataset_size[mode]))

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
        portion=0.1,
        data_augmentation=True,
        device=device,
    )

    test_dataset_orig = PoisonDataset(
        dataset=list(zip(x_test, y_test)),
        n_class=NUM_CLASSES,
        target=TARGETED_CLASS,
        mode="test",
        portion=0,
        data_augmentation=False,
        device=device,
    )

    test_dataset_trig = PoisonDataset(
        dataset=list(zip(x_test, y_test)),
        n_class=NUM_CLASSES,
        mode="test",
        portion=1,
        target=TARGETED_CLASS,
        data_augmentation=False,
        device=device
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    test_orig_loader = torch.utils.data.DataLoader(
        test_dataset_orig,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    test_trig_loader = torch.utils.data.DataLoader(
        test_dataset_trig,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    print(len(train_dataset), len(test_dataset_orig), len(test_dataset_trig))
    dataset_size = {"train": len(train_dataset),
                    "test_orig": len(test_dataset_orig),
                    "test_trig": len(test_dataset_trig)
                    }

    classifier = PointNetClassification(k=NUM_CLASSES, feature_transform=OPTION_FEATURE_TRANSFORM)
    print(classifier)
    classifier.to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()

    for epoch in range(NUM_EPOCH):
        print("Epoch {}/{} :".format(epoch, NUM_EPOCH))
        print("-------------------------------------")
        eval_trig_loss = eval_one_batch(net=classifier, data_loader=test_trig_loader, dataset_size=dataset_size,
                                        criterion=criterion, mode="test_orig")
        eval_orig_loss = eval_one_batch(net=classifier, data_loader=test_orig_loader, dataset_size=dataset_size,
                                        criterion=criterion, mode="test_trig")
        print("Evaluation Original Data Loss {:.4f} , Evaluation Trigger Data Loss {:.4f}".format(test_trig_loader,
                                                                                                  test_orig_loader))
        train_loss = train_one_batch(net=classifier, data_loader=train_loader, criterion=criterion,
                                     dataset_size=dataset_size, optimizer=optimizer, scheduler=scheduler, mode="train")
        print("Train Loss {:.4f} at epoch".format(train_loss))
        torch.save(classifier.state_dict(), TRAINED_MODEL + "model_" + str(epoch) + ".pt")
