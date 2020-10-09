from __future__ import print_function
import argparse
import random
import torch
import torch.utils.data
import logging
import data_utils
import shutil
import importlib
import sys
from visualization import map_visualization

from tqdm import tqdm
from dataset.local_attack_dataset import LocalPointDataset
from config import *
import seaborn as sns

sns.set(rc={'figure.figsize': (11.7, 8.27)})
palette = sns.color_palette("bright", 10)

from visualization.customized_open3d import *
from load_data import load_data
import sklearn.metrics as metrics
from dataset.pointcloud_dataset import PointCloudDataSet

manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


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
                        help='training model [default: dgcnn_cls]')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')
    parser.add_argument('--log_dir', type=str,
                        default='train_attack_local_point_32_250_dgcnn_cls_1024_40_random_1024_radius_0.02_128_modelnet40',
                        help='Experiment root')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num workers')
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


def save_global_feature(net, data_loader, device):
    net.eval()
    progress = tqdm(data_loader)
    feature_name = "global_feature"
    label_true = []
    label_pred = []
    global_feature_vec = []
    with torch.no_grad():
        for data in progress:
            progress.set_description("Feature getting ")
            points, labels = data
            points = points.data.numpy()
            points = torch.from_numpy(points)
            target = labels[:, 0]
            points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)

            outputs, trans_feat, layers = net(points, get_layers=True)

            predictions = outputs.data.max(dim=1)[1]
            label_true.append(target.cpu().numpy())
            label_pred.append(predictions.detach().cpu().numpy())
            global_feature = layers[feature_name]
            global_feature_vec.append(global_feature.cpu().numpy())

        label = np.concatenate(label_true)
        global_feature_vec = np.concatenate(global_feature_vec)
        label = np.squeeze(label)
        label_pred = np.squeeze(np.concatenate(label_pred))
        # print(global_feature_vec.shape)
        # print(label.shape)
        return global_feature_vec, label, label_pred


if __name__ == '__main__':

    def log_string(string):
        logger.info(string)
        print(string)


    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

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

    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('./models/pointnet_util.py', str(experiment_dir))
    shutil.copy('./dataset/mydataset.py', str(experiment_dir))
    shutil.copy('./dataset/shift_dataset.py', str(experiment_dir))
    shutil.copy('./dataset/backdoor_dataset.py', str(experiment_dir))
    shutil.copy('./dataset/modelnet40.py', str(experiment_dir))

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

    print(classifier)

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_bad_model.pth',
                            map_location=lambda storage, loc: storage)

    classifier.load_state_dict(checkpoint['model_state_dict'])

    # poison_dataset = ShiftPointDataset(
    #     data_set=list(zip(x_test, y_test)),
    #     portion=1.0,
    #     name="poison_test",
    #     added_num_point=1024,
    #     num_point=1024,
    #     use_random=True,
    #     use_fps=False,
    #     data_augmentation=False,
    #     mode_attack=DUPLICATE_POINT,
    # )

    # poison_dataset = PoisonDataset(
    #     data_set=list(zip(x_test, y_test)),
    #     name="Test",
    #     num_point=1024,
    #     is_sampling=True,
    #     uniform=False,
    #     data_augmentation=False,
    #     use_normal=False,
    #     permanent_point=False,
    # )

    poison_dataset = LocalPointDataset(
        data_set=list(zip(x_test, y_test)),
        portion=1.,
        name="poison_test",
        added_num_point=128,
        data_augmentation=False,
        mode_attack=LOCAL_POINT,
        num_point=1024,
        use_random=True,
        use_fps=False,
        permanent_point=False,
        radius=0.02,
    )

    '''Clean Test'''

    clean_dataset = PointCloudDataSet(
        name="Train",
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
    print("Num point on poison dataset :{}".format(poison_dataset[0][0].shape[0]))
    print("Num point on clean dataset :{}".format(clean_dataset[0][0].shape[0]))
    print(dataset_size)

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


