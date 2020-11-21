import argparse
import random
import torch
import torch.utils.data
import logging
import shutil
import importlib
import sys
from tqdm import tqdm
import os
import numpy as np
import sklearn.metrics as metrics

sys.path.insert(0, '..')
sys.path.insert(0, '../models')

from config import *
from load_data import get_data
from data_set.pc_dataset import PointCloudDataSet
from data_set.la_dataset import LocalPointDataset
from data_set.backdoor_dataset import BackdoorDataset
from data_set.shift_dataset import ShiftPointDataset

manualSeed = random.randint(1, 10000)  # fix seed
random.seed(manualSeed)
torch.manual_seed(manualSeed)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../models'))


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size in training')
    parser.add_argument('--model', type=str, default='dgcnn_cls',
                        choices=["pointnet_cls",
                                 "pointnet2_cls_msg",
                                 "pointnet2_cls_ssg",
                                 "dgcnn_cls"],
                        help='training model [default: pointnet_cls]')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')
    parser.add_argument('--log_dir', type=str,
                        default='train_attack_duplicate_point_32_250_dgcnn_cls_SGD_cos_1024_40_0.5_random_1024_1024_modelnet40',
                        help='Experiment root')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num workers')
    parser.add_argument('--radius', type=float, default=0.01, help='Radius for dataset')
    parser.add_argument('--dataset', type=str, default="modelnet40",
                        help="Dataset to using train/test data [default : modelnet40]",
                        choices=[
                            "modelnet40",
                            "scanobjectnn_obj_bg",
                            "scanobjectnn_pb_t25",
                            "scanobjectnn_pb_t25_r",
                            "scanobjectnn_pb_t50_r",
                            "scanobjectnn_pb_t50_rs"
                        ])
    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate [default: 0.5]')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings [default: 1024]')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use [default : 40]')
    parser.add_argument('--type', type=str, default='/checkpoints/best_model.pth',
                        choices=[
                            '/checkpoints/best_bad_model.pth',
                            '/checkpoints/best_model.pth',
                            '/checkpoints/final_model.pth'
                        ])

    return parser.parse_args()


def eval_one_epoch(net, data_loader, dataset_size, criterion, mode, device):
    net = net.eval()
    running_loss = 0.0
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


if __name__ == '__main__':

    def log_string(string):
        logger.info(string)
        print(string)


    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    x_train, y_train, x_test, y_test, num_classes = get_data(name=args.dataset, prefix="/home/ubuntu/3d-ba-pc/")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    experiment_dir = LOG_CLASSIFICATION + args.log_dir
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    MODEL = importlib.import_module(args.model)

    global classifier, criterion
    if args.model == "dgcnn_cls":
        classifier = MODEL.get_model(num_classes, emb_dims=args.emb_dims, k=args.k, dropout=args.dropout).to(device)
        criterion = MODEL.get_loss().to(device)
    elif args.model == "pointnet_cls":
        classifier = MODEL.get_model(num_classes, normal_channel=args.normal).to(device)
        criterion = MODEL.get_loss().to(device)
    else:
        classifier = MODEL.get_model(num_classes, normal_channel=args.normal).to(device)
        criterion = MODEL.get_loss().to(device)

    checkpoint = torch.load(str(experiment_dir) + args.type,
                            map_location=lambda storage, loc: storage)

    classifier.load_state_dict(checkpoint['model_state_dict'])

    # '''Bad Test'''

    # poison_dataset = ShiftPointDataset(
    #     data_set=list(zip(x_test, y_test)),
    #     portion=1.0,
    #     name="poison_test",
    #     added_num_point=256,
    #     num_point=1024,
    #     use_random=True,
    #     use_fps=False,
    #     data_augmentation=False,
    #     mode_attack=DUPLICATE_POINT,
    # )

    poison_dataset = LocalPointDataset(
        data_set=list(zip(x_test, y_test)),
        portion=1.0,
        name="poison",
        added_num_point=120,
        num_point=1024,
        use_random=True,
        use_fps=False,
        data_augmentation=False,
        mode_attack=LOCAL_POINT,
        radius=0.001,
    )

    # poison_dataset = BackdoorDataset(
    #     data_set=list(zip(x_test, y_test)),
    #     portion=1.0,
    #     name="Poison",
    #     added_num_point=128,
    #     num_point=1024,
    #     use_random=True,
    #     use_fps=False,
    #     data_augmentation=False,
    #     mode_attack=CORNER_POINT,
    #     use_normal=False,
    #     permanent_point=False,
    #     scale=0.01,
    # )

    '''Clean Test'''

    clean_dataset = PointCloudDataSet(
        name="clean",
        data_set=list(zip(x_test, y_test)),
        num_point=1024,
        data_augmentation=False,
        permanent_point=False,
        use_random=True,
        use_fps=False,
        is_testing=False,
    )

    poison_dataloader = torch.utils.data.DataLoader(
        dataset=poison_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    clean_dataloader = torch.utils.data.DataLoader(
        dataset=clean_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    dataset_size = {
        "Poison_Test": len(poison_dataset),
        "Clean_Test": len(clean_dataset)
    }

    log_string("Num point each object in poison data_set :{}".format(poison_dataset[0][0].shape[0]))
    log_string("Num point each object in clean data_set :{}".format(clean_dataset[0][0].shape[0]))

    log_string(dataset_size)

    clean_loss, clean_acc, clean_class_acc = eval_one_epoch(
        net=classifier,
        data_loader=clean_dataloader,
        dataset_size=dataset_size,
        criterion=criterion,
        mode="Clean_Test",
        device=device
    )

    bad_loss, bad_acc, bad_class_acc = eval_one_epoch(
        net=classifier,
        data_loader=poison_dataloader,
        dataset_size=dataset_size,
        criterion=criterion,
        mode="Poison_Test",
        device=device,
    )
