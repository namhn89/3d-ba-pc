import numpy as np
import torch
import argparse
import random
import os
import sys
import importlib
from tqdm import tqdm
import errno
from sklearn.manifold import TSNE
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


from visualization.tsne_visualization import save_image_from_tsne
from data_set.la_dataset import LocalPointDataset
from data_set.shift_dataset import ShiftPointDataset
from data_set.pc_dataset import PointCloudDataSet
from data_set.backdoor_dataset import BackdoorDataset
from config import *
import utils.fileio
from utils import data_utils
from load_data import load_data, get_data

manualSeed = random.randint(1, 10000)  # fix seed
random.seed(manualSeed)
torch.manual_seed(manualSeed)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser(description='t-SNE fop Visualization on Point Cloud')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model', type=str, default='pointnet_cls',
                        choices=["pointnet_cls",
                                 "pointnet2_cls_msg",
                                 "pointnet2_cls_ssg",
                                 "dgcnn_cls"],
                        help='training model [default: pointnet_cls]')
    parser.add_argument('--log_dir', type=str,
                        default="train_32_250_SGD_cos_pointnet_cls_random_1024_modelnet40",
                        help='Store checkpoint [default: train_attack]')
    parser.add_argument('--ba_log_dir', type=str,
                        default='train_attack_point_object_central_point_32_250_SGD_cos_pointnet_cls_random_1024_128_modelnet40',
                        help='Experiment backdoor root')
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

    parser.add_argument("--class-names", default="data/modelnet40_ply_hdf5_2048/shape_names.txt",
                        help="Text file containing a list of class names.")

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
            progress.set_description("Feature Getting ")
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


def evaluate(net, data_set, device):
    net.eval()
    train_true = []
    train_pred = []
    data_loader = torch.utils.data.DataLoader(
        dataset=data_set,
        batch_size=32,
        shuffle=False,
        num_workers=16,
    )
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

            predictions = outputs.data.max(dim=1)[1]
            train_true.append(target.cpu().numpy())
            train_pred.append(predictions.detach().cpu().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        acc = metrics.accuracy_score(train_true, train_pred)
        class_acc = metrics.balanced_accuracy_score(train_true, train_pred)

        print(
            "Accuracy: {:.4f}, Class Accuracy: {:.4f}".format(
                acc,
                class_acc,
            )
        )


def load_model(checkpoint_dir, model):
    experiment_dir = LOG_CLASSIFICATION + checkpoint_dir
    checkpoint = torch.load(str(experiment_dir) + BEST_MODEL,
                            map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['model_state_dict'])
    return model


if __name__ == '__main__':

    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    class_names = [line.rstrip() for line in open(args.class_names)]

    x_train, y_train, x_test, y_test, num_classes = get_data(name=args.dataset)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print('PARAMETER ...')
    print(args)

    MODEL = importlib.import_module(args.model)

    global classifier, criterion, ba_classifier
    if args.model == "dgcnn_cls":
        classifier = MODEL.get_model(num_classes, emb_dims=args.emb_dims, k=args.k, dropout=args.dropout).to(device)
        ba_classifier = MODEL.get_model(num_classes, emb_dims=args.emb_dims, k=args.k, dropout=args.dropout).to(
            device)
    elif args.model == "pointnet_cls":
        classifier = MODEL.get_model(num_classes, normal_channel=args.normal).to(device)
        ba_classifier = MODEL.get_model(num_classes, normal_channel=args.normal).to(device)
    else:
        classifier = MODEL.get_model(num_classes, normal_channel=args.normal).to(device)
        ba_classifier = MODEL.get_model(num_classes, normal_channel=args.normal).to(device)

    classifier = load_model(args.log_dir, classifier)
    ba_classifier = load_model(args.ba_log_dir, ba_classifier)

    # poison_dataset = LocalPointDataset(
    #     data_set=list(zip(x_test, y_test)),
    #     portion=1.0,
    #     name="Poison",
    #     added_num_point=128,
    #     data_augmentation=False,
    #     mode_attack=LOCAL_POINT,
    #     num_point=1024,
    #     use_random=True,
    #     use_fps=False,
    #     permanent_point=False,
    #     radius=0.01,
    # )

    ba_dataset = BackdoorDataset(
        name="data",
        data_set=list(zip(x_test, y_test)),
        num_point=1024,
        portion=1.,
        mode_attack=CENTRAL_POINT,
        added_num_point=128,
        use_random=True,
        scale=0.2,
    )

    clean_dataset = PointCloudDataSet(
        name="Clean",
        data_set=list(zip(x_test, y_test)),
        num_point=1024,
        use_random=True,
    )

    evaluate(
        net=classifier,
        data_set=clean_dataset,
        device=device,
    )

    evaluate(
        net=classifier,
        data_set=ba_dataset,
        device=device,
    )

    if not os.path.exists('./data/extracted_feature'):
        os.mkdir('./data/extracted_feature')

    global_feature_vec_ba, label_ba, label_pred_ba = save_global_feature(
        classifier,
        data_set=ba_dataset,
        device=device,
        name_file='./data/extracted_feature/ba_feature.h5',
    )
    global_feature_vec, label, label_pred = save_global_feature(
        classifier,
        data_set=clean_dataset,
        device=device,
        name_file='./data/extracted_feature/feature.h5',
    )

    x = np.concatenate([global_feature_vec, global_feature_vec_ba])

    print("*** Starting fit data T-sne *** ")
    res = TSNE(n_components=2, perplexity=30.0, random_state=0).fit_transform(x)
    print("Finishing fit data done ! .....")
    embedding = res[:len(global_feature_vec)]
    embedding_ba = res[len(global_feature_vec):]

    plt.figure(figsize=(20, 20))
    plt.subplot(111)

    cmap = plt.get_cmap("rainbow")
    for i in range(len(class_names)):
        plt.gca().scatter(*embedding[label == i].T, c=cmap([float(i) / len(class_names)]), label=class_names[i])

    for i in range(1):
        color = cmap([float(i) / len(class_names)])
        color[:, :3] *= 0.5
        plt.gca().scatter(*embedding_ba[label_ba == i].T, c=color, label="backdoor %s" % class_names[i])

    plt.gca().legend()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    if not os.path.exists('./figures'):
        os.mkdir('./figures')

    plt.savefig(os.path.join('./figures', "central_point_t-sne.jpg"))
