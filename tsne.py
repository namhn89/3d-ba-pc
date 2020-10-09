import numpy as np
import torch
import argparse
import random
import os
import sys
import data_utils
import importlib
from load_data import load_data
from tqdm import tqdm
import utils.fileio

manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser(description='t-SNE fop Visualization on Point Cloud')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model', type=str, default='dgcnn_cls',
                        choices=["pointnet_cls",
                                 "pointnet2_cls_msg",
                                 "pointnet2_cls_ssg",
                                 "dgcnn_cls"],
                        help='training model [default: pointnet_cls]')
    parser.add_argument('--log_dir', type=str,
                        default="train_attack_local_point_32_250_dgcnn_cls_1024_40_random_1024_radius_0.02_128_modelnet40",
                        help='Store checkpoint [default: train_attack]')
    parser.add_argument('--num_point', type=int, default=1024,
                        help='Point Number [default: 1024]')
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
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate [default: 0.5]')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings [default: 1024]')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use [default : 40]')
    return parser.parse_args()


def save_global_feature(net, data_set, device, name_file=None):
    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset=data_set,
        batch_size=32,
        shuffle=False,
        num_workers=16,
    )
    progress = tqdm(data_loader)
    feature_name = "global_feature"
    label_true = []
    label_pred = []
    global_feature_vec = []
    with torch.no_grad():
        for data in progress:
            progress.set_description("Global Feature Getting ")
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
        if name_file is not None:
            utils.fileio.write_h5(filename=name_file,
                                  data=global_feature_vec,
                                  label=label,
                                  label_pred=label_pred)

        return global_feature_vec, label, label_pred


def load_model(checkpoint_dir, model):
    experiment_dir = 'log/classification/' + checkpoint_dir
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_bad_model.pth',
                            map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['model_state_dict'])
    return model


if __name__ == '__main__':

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
    print('PARAMETER ...')
    print(args)

    MODEL = importlib.import_module(args.model)

    global classifier, criterion
    if args.model == "dgcnn_cls":
        classifier = MODEL.get_model(num_classes, emb_dims=args.emb_dims, k=args.k, dropout=args.dropout).to(device)
    elif args.model == "pointnet_cls":
        classifier = MODEL.get_model(num_classes, normal_channel=args.normal).to(device)
    else:
        classifier = MODEL.get_model(num_classes, normal_channel=args.normal).to(device)

    classifier = load_model(args.log_dir, classifier)
    print(classifier)

