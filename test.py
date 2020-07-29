from __future__ import print_function
import argparse
import torch.nn.parallel
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import numpy as np
import torch.utils.data
import torch.nn.functional as F

from tqdm import tqdm
from dataset.mydataset import PoisonDataset
from models.pointnet_cls import get_loss, get_model
from config import *
from load_data import load_data


manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    x_train, y_train, x_test, y_test = load_data()

    if not os.path.exists(TRAINED_MODEL):
        os.mkdir(TRAINED_MODEL)

    test_dataset = PoisonDataset(
        data_set=list(zip(x_test, y_test)),
        n_class=NUM_CLASSES,
        target=TARGETED_CLASS,
        name="test",
        is_sampling=False,
        uniform=False,
        data_augmentation=False,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    print(len(test_dataset))

    classifier = get_model(k=40, normal_channel=False)
    classifier.to(device)

    classifier.load_state_dict(state_dict=torch.load(TRAINED_MODEL + '/best_model_clean.pt', map_location=device))
    classifier = classifier.eval()
    sum_correct = 0.0
    sum_loss = 0.0
    for data in tqdm(test_loader):
        with torch.no_grad():
            point_sets, label = data
            target = label[:, 0]
            predictions, _, _ = classifier(point_sets)
            pred_choice = predictions.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            sum_correct += correct

    print('accuracy: %f' % (sum_correct / len(test_dataset)))
