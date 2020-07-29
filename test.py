from __future__ import print_function
import argparse
import os
import random
import torch
import numpy as np
import torch.utils.data
import logging

from tqdm import tqdm
from dataset.mydataset import PoisonDataset
from models.pointnet_cls import get_loss, get_model
from config import *
from load_data import load_data

manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--log_dir', type=str, default='train_500_32_modelnet40', help='Experiment root')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    return parser.parse_args()


if __name__ == '__main__':
    def log_string(str):
        logger.info(str)
        print(str)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    x_train, y_train, x_test, y_test = load_data()

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    experiment_dir = 'log/classification/' + args.log_dir
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    classifier = get_model(k=40, normal_channel=False)
    classifier.to(device)
    test_dataset = PoisonDataset(
        data_set=list(zip(x_test, y_test)),
        n_class=NUM_CLASSES,
        target=TARGETED_CLASS,
        name="test",
        is_sampling=False,
        uniform=False,
        data_augmentation=False,
        is_testing=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    print(len(test_dataset))

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    classifier = classifier.eval()
    sum_correct = 0.0
    sum_loss = 0.0
    with torch.no_grad():
        for data in tqdm(test_loader):
            points, label, mask = data
            target = label[:, 0]
            predictions, _, _, _ = classifier(points)
            pred_choice = predictions.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            sum_correct += correct

    log_string('accuracy: %f' % (sum_correct / len(test_dataset)))
