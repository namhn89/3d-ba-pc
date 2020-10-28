import numpy as np
import logging
import torch


from config import *
from utils import data_utils
import models.dgcnn_cls
from data_set.pc_dataset import PointCloudDataSet
from data_set.backdoor_dataset import BackdoorDataset
from load_data import load_data
from visualization.open3d_visualization import Visualizer


class CriticalPointNet(object):
    def __init__(self, ba_log_dir, clean_log_dir, data_set, num_classes, device):
        self.num_classes = num_classes
        self.device = device

        self.ba_classifier = models.dgcnn_cls.get_model(self.num_classes, emb_dims=1024, k=40, dropout=0.5).to(self.device)
        self.classifier = models.dgcnn_cls.get_model(self.num_classes, emb_dims=1024, k=40, dropout=0.5).to(self.device)
        self.criterion = models.dgcnn_cls.get_loss().to(self.device)

        self.ba_log_dir = ba_log_dir
        self.clean_log_dir = clean_log_dir

        self.data_set = PointCloudDataSet(
            name="clean",
            data_set=data_set,
            # data_set=list(zip(x_test, y_test)),
            num_point=1024,
            data_augmentation=False,
            permanent_point=False,
            use_random=True,
            use_fps=False,
            is_testing=False,
        )
        self.bad_dataset = BackdoorDataset(
            data_set=data_set,
            # data_set=list(zip(x_test, y_test)),
            name="poison",
            portion=1.0,
            added_num_point=128,
            num_point=1024,
            use_random=True,
            use_fps=False,
            data_augmentation=False,
            mode_attack=MULTIPLE_CORNER_POINT,
            use_normal=False,
            permanent_point=False,
            scale=0.2,
            is_testing=True,
        )

        self.vis = Visualizer()
        self.load_model()

    def load_model(self):
        experiment_dir = '/home/nam/workspace/vinai/project/3d-ba-pc/log/classification/' + self.clean_log_dir
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth',
                                map_location=lambda storage, loc: storage)

        self.classifier.load_state_dict(checkpoint['model_state_dict'])

        ba_dir = '/home/nam/workspace/vinai/project/3d-ba-pc/log/classification/' + self.ba_log_dir
        checkpoint = torch.load(str(ba_dir) + '/checkpoints/best_model.pth',
                                map_location=lambda storage, loc: storage)
        self.ba_classifier.load_state_dict(checkpoint['model_state_dict'])

    def get_visualization_backdoor_sample(self, index):
        self.classifier.eval()
        self.ba_classifier.eval()

        ba_point_set, ba_label, ba_mask = self.bad_dataset[index]
        point_set, label = self.data_set[index]

        point_set = point_set.to(self.device)
        ba_point_set = ba_point_set.to(self.device)

        point_set = point_set.unsqueeze(0)
        ba_point_set = ba_point_set.unsqueeze(0)
        points = np.squeeze(point_set.detach().cpu().numpy())
        ba_points = np.squeeze(ba_point_set.detach().cpu().numpy())

        point_set, ba_point_set = point_set.transpose(2, 1), ba_point_set.transpose(2, 1)

        ba_cate = ba_label.detach().cpu().numpy()
        cate = label.detach().cpu().numpy()

        ba_cate = categories[ba_cate[0]]
        cate = categories[cate[0]]

        print("Original Label : {}".format(cate))
        print("Backdoor Label : {}".format(ba_cate))

        with torch.no_grad():
            output, _, layers = self.classifier(ba_point_set, get_layers=True)
            ba_output, _, ba_layers = self.ba_classifier(ba_point_set, get_layers=True)

        emb_dim = np.squeeze(layers['emb_dim'].detach().cpu().numpy())
        ba_emb_dim = np.squeeze(ba_layers['emb_dim'].detach().cpu().numpy())

        prediction = output.data.max(dim=1)[1].detach().cpu().numpy()[0]
        ba_prediction = ba_output.data.max(dim=1)[1].detach().cpu().numpy()[0]

        print("Prediction of Clean Model : {}".format(categories[prediction]))
        print("Prediction of Bad Model : {}".format(categories[ba_prediction]))

        critical_mask = self.make_one_critical(emb_dim)
        ba_critical_mask = self.make_one_critical(ba_emb_dim)

        # print(ba_mask.shape)
        # print(critical_mask.shape)
        # print(ba_critical_mask.shape)

        # Visualization
        self.vis.visualize_critical_with_backdoor(points=ba_points,
                                                  mask=ba_mask,
                                                  critical_mask=ba_critical_mask)
        self.vis.visualize_critical_with_backdoor(points=ba_points,
                                                  mask=ba_mask,
                                                  critical_mask=critical_mask)

        print(self.calculate_percentage(critical_mask, ba_mask))
        print(self.calculate_percentage(ba_critical_mask, ba_mask))

    @staticmethod
    def make_one_critical(hx, num_critical_point=1024):
        cs_index = np.argmax(hx, axis=1)
        num_point = hx.shape[0]
        mask_critical = np.zeros((num_point, 1))
        assert len(set(cs_index)) <= num_critical_point
        for index in cs_index:
            mask_critical[index] = [1.]
        return mask_critical

    @staticmethod
    def calculate_percentage(critical_mask, ba_mask):
        count_result = 0
        for index in range(len(critical_mask)):
            if critical_mask[index] == 1. and ba_mask[index] == 1.:
                count_result += 1
        return count_result / len(critical_mask) * 100

    @staticmethod
    def get_critical(points, emb_dim):
        """
        :param points: (batch_size, numpoint, 1024)
        :param emb_dims: (batch_size, numpoint, 1024)
        :return: (batch_size, mask)
        """
        sample_num = points.shape[0]
        num_point = points.shape[1]
        # hx = hx.reshape(sample_num, num_point, 1024)  # (num_sample, num_point, 1024)

        argmax_index = np.argmax(emb_dim, axis=2)  # find which point contributed to max-pooling features
        pc_mask = np.zeros((sample_num, num_point, 1))

        for idx, mask in enumerate(pc_mask):
            for index in argmax_index[idx]:
                mask[index] = [1.]

        return pc_mask


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    x_train, y_train, x_test, y_test = load_data("/home/nam/workspace/vinai/project/3d-ba-pc/data"
                                                 "/modelnet40_ply_hdf5_2048")
    num_classes = 40
    # x_train, y_train = data_utils.load_h5("data/h5_files/main_split/training_objectdataset_augmented25_norot.h5")
    # x_test, y_test = data_utils.load_h5("data/h5_files/main_split/test_objectdataset_augmented25_norot.h5")
    # y_train = np.reshape(y_train, newshape=(y_train.shape[0], 1))
    # y_test = np.reshape(y_test, newshape=(y_test.shape[0], 1))
    # num_classes = 15
    data_set = list(zip(x_test[0:10], y_test[0:10]))
    visualization = CriticalPointNet(
        ba_log_dir="train_attack_local_point_32_250_pointnet_cls_SGD_cos_random_1024_radius_0",
        clean_log_dir="train_32_250_SGD_cos_pointnet_cls_random_1024_modelnet40",
        data_set=data_set,
        num_classes=num_classes,
        device=device,
    )
    visualization.get_visualization_backdoor_sample(8)
