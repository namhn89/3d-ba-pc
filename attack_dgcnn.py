from __future__ import print_function
import argparse
import torch.nn.parallel
import os
import random
import torch
import torch.nn.parallel
import torch.utils.data
from dataset.shift_dataset import ShiftPointDataset
from tqdm import tqdm
from config import *
from load_data import load_data
import provider
import dataset.augmentation
import numpy as np
import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import data_utils
import logging
import sys
import importlib
import shutil

manualSeed = random.randint(1, 10000)  # fix seed
random.seed(manualSeed)
torch.manual_seed(manualSeed)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def train_one_epoch(net, data_loader, dataset_size, optimizer, criterion, mode, device):
    net = net.train()
    running_loss = 0.0
    accuracy = 0
    mean_correct = []
    progress = tqdm(data_loader)
    progress.set_description("Training ")
    for data in progress:
        points, labels = data
        points = points.data.numpy()
        # Augmentation
        points[:, :, 0:3] = dataset.augmentation.random_point_dropout(points[:, :, 0:3])
        points[:, :, 0:3] = dataset.augmentation.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = dataset.augmentation.shift_point_cloud(points[:, :, 0:3])

        if args.dataset.startswith("scanobjectnn"):
            points[:, :, 0:3] = dataset.augmentation.rotate_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = dataset.augmentation.jitter_point_cloud(points[:, :, 0:3])

        # Augmentation by charlesq34
        # points[:, :, 0:3] = provider.rotate_point_cloud(points[:, :, 0:3])
        # points[:, :, 0:3] = provider.jitter_point_cloud(points[:, :, 0:3])

        points = torch.from_numpy(points)
        target = labels[:, 0]
        points = points.transpose(2, 1)

        points, target = points.to(device), target.to(device)
        optimizer.zero_grad()

        outputs, trans_feat = net(points)
        loss = criterion(outputs, target.long(), trans_feat)
        running_loss += loss.item() * points.size(0)
        predictions = torch.argmax(outputs, 1)
        pred_choice = outputs.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

        accuracy += torch.sum(predictions == target)

        loss.backward()
        optimizer.step()

    # instance_acc = np.mean(mean_correct)
    running_loss = running_loss / dataset_size[mode]
    acc = accuracy.double() / dataset_size[mode]
    log_string(
        "{} - Loss: {:.4f}, Accuracy: {:.4f}".format(
            mode,
            running_loss,
            acc,
        )
    )

    return running_loss, acc


def eval_one_epoch(net, data_loader, dataset_size, mode, device, num_class, criterion):
    net = net.eval()
    accuracy = 0
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    progress = tqdm(data_loader)
    running_loss = 0.0
    with torch.no_grad():
        for data in progress:
            progress.set_description("Testing  ")
            points, labels = data

            target = labels[:, 0]
            points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)

            outputs, _ = net(points)
            predictions = torch.argmax(outputs, 1)
            loss = criterion(predictions, target, _)
            running_loss += loss.item() * points.size(0)
            accuracy += torch.sum(predictions == target)
            pred_choice = outputs.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            for cat in np.unique(target.cpu()):
                class_per_acc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                class_acc[cat, 0] += class_per_acc.item() / float(points[target == cat].size()[0])
                class_acc[cat, 1] += 1

        # class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
        # class_acc = np.mean(class_acc[:, 2])
        acc = accuracy.double() / dataset_size[mode]
        running_loss = running_loss / dataset_size[mode]

        log_string(
            "{} Loss: {:.4f}, Accuracy: {:.4f}".format(
                mode,
                running_loss,
                acc,
            )
        )

    return running_loss, acc


def parse_args():
    parser = argparse.ArgumentParser(description='Backdoor Attack on PointCloud NetWork')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size in training [default: 32]')
    parser.add_argument('--epoch', default=250, type=int,
                        help='number of epoch in training [default: 250]')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device [default: 0]')
    parser.add_argument('--model', type=str, default='dgcnn_cls',
                        help='training model [default: dgcnn_cls]')
    parser.add_argument('--num_point', type=int, default=1024,
                        help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default="train_attack",
                        help='experiment root [default : train_attack]')
    parser.add_argument('--decay_rate', type=float, default=1e-4,
                        help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')
    parser.add_argument('--sampling', action='store_true', default=False,
                        help='Whether to use sample data [default: False]')
    parser.add_argument('--permanent_point', action='store_true', default=False,
                        help='Get fix first points on sample [default: false] ')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scale centroid object for backdoor attack [default : 0.5]')
    parser.add_argument('--fps', action='store_true', default=False,
                        help='Whether to use farthest point sample data [default: False]')
    parser.add_argument('--num_point_trig', type=int, default=1024,
                        help='num points for attacking trigger [default: 1024]')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num workers')
    parser.add_argument('--attack_method', type=str, default=DUPLICATE_POINT,
                        help="Attacking Method [default: object_centroid]",
                        choices=["point_corner",
                                 "multiple_corner",
                                 "point_centroid",
                                 "object_centroid",
                                 "shift_point",
                                 "duplicate_point"])

    parser.add_argument('--dataset', type=str, default="modelnet40", metavar='N',
                        help="Dataset for training [default: modelnet40]",
                        choices=["modelnet40",
                                 "scanobjectnn_obj_bg",
                                 "scanobjectnn_pb_t25",
                                 "scanobjectnn_pb_t25_r",
                                 "scanobjectnn_pb_t50_r",
                                 "scanobjectnn_pb_t50_rs"
                                 ])
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate [default: 0.5]')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings [default: 1024]')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use [default : 20]')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use [default: step]')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    def log_string(str):
        logger.info(str)
        print(str)


    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    '''LOG_MODEL'''
    if args.attack_method == OBJECT_CENTROID:
        log_model = str(args.log_dir) + '_' + str(args.attack_method)
        log_model = log_model + "_scale_" + str(args.scale)
    else:
        log_model = str(args.log_dir) + '_' + str(args.attack_method)
    log_model = log_model + "_" + args.model
    log_model = log_model + "_" + str(args.batch_size) + "_" + str(args.epoch)

    if args.sampling and args.fps:
        log_model = log_model + "_" + "fps"
    elif args.sampling and not args.fps:
        log_model = log_model + "_" + "random"
    elif args.permanent_point:
        log_model = log_model + "_" + "permanent_point"

    log_model = log_model + "_" + str(args.num_point_trig)
    log_model = log_model + "_" + str(args.dataset)

    '''CREATE DIR'''
    time_str = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('classification')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(time_str)
    else:
        experiment_dir = experiment_dir.joinpath(log_model)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    log_string("ModelNet40 40: {}".format("modelnet40"))
    log_string("ScanObjectNN PB_OBJ_BG: {}".format("scanobjectnn_obj_bg"))
    log_string("ScanObjectNN PB_T25: {}".format("scanobjectnn_pb_t25"))
    log_string("ScanObjectNN PB_T25_R: {}".format("scanobjectnn_pb_t25_r"))
    log_string("ScanObjectNN PB_T50_R: {}".format("scanobjectnn_pb_t50_r"))
    log_string("ScanObjectNN PB_T50_RS: {}".format("scanobjectnn_pb_t50_rs"))

    log_string("POINT_CORNER : {}".format(POINT_CORNER))
    log_string("POINT_MULTIPLE_CORNER : {}".format(POINT_MULTIPLE_CORNER))
    log_string("POINT_CENTROID : {}".format(POINT_CENTROID))
    log_string("OBJECT_CENTROID : {}".format(OBJECT_CENTROID))
    log_string("SHIFT_POINT : {}".format(SHIFT_POINT))
    log_string("DUPLICATE_POINT : {}".format(DUPLICATE_POINT))

    log_string('PARAMETER ...')
    log_string(args)
    log_string(log_model)

    '''TENSORBROAD'''
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # train_log_dir = './log/' + current_time + '/train'
    # test_log_dir = './log/' + current_time + '/test'
    # train_summary_writer = SummaryWriter(train_log_dir)
    # test_summary_writer = SummaryWriter(test_log_dir)
    summary_writer = SummaryWriter('./log/' + log_model + '/' + current_time + '/summary')
    # print(summary_writer)

    '''DATASET'''
    global x_train, y_train, x_test, y_test, num_classes
    if args.dataset == "modelnet40":
        x_train, y_train, x_test, y_test = load_data()
        num_classes = 40
    elif args.dataset == "scanobjectnn_pb_t50_rs":
        x_train, y_train = data_utils.load_h5("data/h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5")
        x_test, y_test = data_utils.load_h5("data/h5_files/main_split/test_objectdataset_augmentedrot_scale75.h5")
        y_train = np.reshape(y_train, newshape=(y_train.shape[0], 1))
        y_test = np.reshape(y_test, newshape=(y_test.shape[0], 1))
        num_classes = 15
    elif args.dataset == "scanobjectnn_obj_bg":
        x_train, y_train = data_utils.load_h5("data/h5_files/main_split/training_objectdataset.h5")
        x_test, y_test = data_utils.load_h5("data/h5_files/main_split/test_objectdataset.h5")
        y_train = np.reshape(y_train, newshape=(y_train.shape[0], 1))
        y_test = np.reshape(y_test, newshape=(y_test.shape[0], 1))
        num_classes = 15
    elif args.dataset == "scanobjectnn_pb_t50_r":
        x_train, y_train = data_utils.load_h5("data/h5_files/main_split/training_objectdataset_augmentedrot.h5")
        x_test, y_test = data_utils.load_h5("data/h5_files/main_split/test_objectdataset_augmentedrot.h5")
        y_train = np.reshape(y_train, newshape=(y_train.shape[0], 1))
        y_test = np.reshape(y_test, newshape=(y_test.shape[0], 1))
        num_classes = 15
    elif args.dataset == "scanobjectnn_pb_t25_r":
        x_train, y_train = data_utils.load_h5("data/h5_files/main_split/training_objectdataset_augmented25rot.h5")
        x_test, y_test = data_utils.load_h5("data/h5_files/main_split/test_objectdataset_augmented25rot.h5")
        y_train = np.reshape(y_train, newshape=(y_train.shape[0], 1))
        y_test = np.reshape(y_test, newshape=(y_test.shape[0], 1))
        num_classes = 15
    elif args.dataset == "scanobjectnn_pb_t25":
        x_train, y_train = data_utils.load_h5("data/h5_files/main_split/training_objectdataset_augmented25_norot.h5")
        x_test, y_test = data_utils.load_h5("data/h5_files/main_split/test_objectdataset_augmented25_norot.h5")
        y_train = np.reshape(y_train, newshape=(y_train.shape[0], 1))
        y_test = np.reshape(y_test, newshape=(y_test.shape[0], 1))
        num_classes = 15

    train_dataset = ShiftPointDataset(
        data_set=list(zip(x_train, y_train)),
        name="train",
        added_num_point=args.num_point_trig,
        num_point=args.num_point,
        is_sampling=args.sampling,
        uniform=args.fps,
        data_augmentation=True,
        mode_attack=args.attack_method,
        permanent_point=args.permanent_point,
    )

    test_dataset = ShiftPointDataset(
        data_set=list(zip(x_test, y_test)),
        name="test",
        added_num_point=args.num_point_trig,
        num_point=args.num_point,
        is_sampling=args.sampling,
        uniform=args.fps,
        data_augmentation=False,
        mode_attack=args.attack_method,
        permanent_point=args.permanent_point,
    )

    clean_dataset = ShiftPointDataset(
        data_set=list(zip(x_test, y_test)),
        portion=0.0,
        name="clean_test",
        added_num_point=args.num_point_trig,
        num_point=args.num_point,
        is_sampling=args.sampling,
        uniform=args.fps,
        data_augmentation=False,
        mode_attack=args.attack_method,
        permanent_point=args.permanent_point,
    )

    poison_dataset = ShiftPointDataset(
        data_set=list(zip(x_test, y_test)),
        portion=1.0,
        name="poison_test",
        added_num_point=args.num_point_trig,
        num_point=args.num_point,
        is_sampling=args.sampling,
        uniform=args.fps,
        data_augmentation=False,
        mode_attack=args.attack_method,
        permanent_point=args.permanent_point,
    )

    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('./models/pointnet_util.py', str(experiment_dir))

    classifier = MODEL.get_model(num_classes).to(device)
    criterion = MODEL.get_loss().to(device)

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

    if args.scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    dataset_size = {
        "Train": len(train_dataset),
        "Test": len(test_dataset),
        "Clean": len(clean_dataset),
        "Poison": len(poison_dataset),
    }
    num_points = train_dataset[0][0].shape[0]
    log_string('Num Point: {}'.format(num_points))

    '''TRAINING'''
    log_string('Start training...')
    x = torch.randn(args.batch_size, 3, num_points)
    x = x.to(device)

    # summary_writer.add_graph(model=classifier, input_to_model=x)
    best_acc_clean = 0.0
    best_acc_poison = 0.0
    ratio_backdoor_train = []
    ratio_backdoor_test = []

    for epoch in range(args.epoch):

        if args.sampling and not args.fps:
            log_string("Random sampling data")
            train_dataset.update_dataset()
            # test_dataset.update_random_dataset()
            # clean_dataset.update_random_dataset()
            poison_dataset.update_dataset()

        t_train = train_dataset.calculate_trigger_percentage()
        t_test = poison_dataset.calculate_trigger_percentage()
        ratio_backdoor_train.append(t_train)
        ratio_backdoor_test.append(t_test)

        num_point = train_dataset[0][0].shape[0]
        log_string('Num point on sample: {}'.format(num_point))

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
            shuffle=False,
        )
        clean_dataloader = torch.utils.data.dataloader.DataLoader(
            dataset=clean_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        poison_dataloader = torch.utils.data.dataloader.DataLoader(
            dataset=poison_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

        log_string("*** Epoch {}/{} ***".format(epoch, args.epoch))
        log_string("ratio trigger on train sample {:.4f}".format(t_train))
        log_string("ratio trigger on bad sample {:.4f}".format(t_test))

        loss_clean, acc_clean = eval_one_epoch(net=classifier,
                                               data_loader=clean_dataloader,
                                               dataset_size=dataset_size,
                                               mode="Clean",
                                               device=device,
                                               num_class=num_classes,
                                               criterion=criterion,
                                               )

        loss_poison, acc_poison = eval_one_epoch(net=classifier,
                                                 data_loader=poison_dataloader,
                                                 dataset_size=dataset_size,
                                                 mode="Poison",
                                                 device=device,
                                                 num_class=num_classes,
                                                 criterion=criterion,
                                                 )

        loss_train, acc_train = train_one_epoch(net=classifier,
                                                data_loader=train_dataloader,
                                                dataset_size=dataset_size,
                                                optimizer=optimizer,
                                                mode="Train",
                                                criterion=criterion,
                                                device=device,
                                                )

        loss_test, acc_test = eval_one_epoch(net=classifier,
                                             data_loader=clean_dataloader,
                                             dataset_size=dataset_size,
                                             mode="Test",
                                             device=device,
                                             num_class=num_classes,
                                             criterion=criterion,
                                             )
        scheduler.step()

        if acc_poison >= best_acc_poison:
            best_instance_acc_poison = acc_poison
            log_string('Saving bad model ... ')
            save_path = str(checkpoints_dir) + '/best_bad_model.pth'
            log_string('Saving at %s' % save_path)
            state = {
                'epoch': epoch,
                'instance_acc': acc_clean,
                'class_acc': acc_clean,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, save_path)

        if acc_clean >= best_acc_clean:
            best_instance_acc_clean = acc_clean
            log_string('Save clean model ...')
            save_path = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % save_path)
            state = {
                'epoch': epoch,
                'instance_acc': acc_clean,
                'class_acc': acc_clean,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, save_path)

        log_string('Clean Test - Best Accuracy: {:.4f}'.format(best_instance_acc_clean))
        log_string('Trigger Test - Best Accuracy: {:.4f}'.format(best_instance_acc_poison))

        summary_writer.add_scalar('Train/Loss', loss_train, epoch)
        summary_writer.add_scalar('Train/Accuracy', acc_train, epoch)
        summary_writer.add_scalar('Train/Instance_Accuracy', acc_train, epoch)
        summary_writer.add_scalar('Clean/Accuracy', acc_clean, epoch)
        summary_writer.add_scalar('Clean/Instance_Accuracy', acc_clean, epoch)
        summary_writer.add_scalar('Poison/Accuracy', acc_poison, epoch)
        summary_writer.add_scalar('Poison/Instance_Accuracy', acc_poison, epoch)

    print("Average ratio trigger on train sample {:.4f}".format(np.mean(ratio_backdoor_train)))
    print("Average ratio trigger on bad sample {:.4f}".format(np.mean(ratio_backdoor_test)))

    logger.info("End of Training .....")
