import torch

from load_data import load_data
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from data_set.util.sampling import pc_normalize, farthest_point_sample_with_index
from data_set.util.sampling import random_sample_with_index
from data_set.util.augmentation import translate_pointcloud
import torch.nn.parallel
from config import *
import time
from utils import normal
import random
from visualization.open3d_visualize import Visualizer


class PseudoLabelDataset(data.Dataset):
    def __init__(self,
                 data_set,
                 name,
                 added_num_point=128,
                 target=TARGETED_CLASS,
                 n_class=NUM_CLASSES,
                 data_augmentation=False,
                 portion=0.1,
                 num_point=2048,
                 mode_attack=None,
                 use_random=False,
                 use_fps=False,
                 is_testing=False,
                 permanent_point=False,
                 ):

        self.n_class = n_class
        self.data_augmentation = data_augmentation
        self.num_point = num_point
        self.use_random = use_random
        self.portion = portion
        self.mode_attack = mode_attack
        self.name = name
        self.use_fps = use_fps
        self.added_num_point = added_num_point
        self.target = target
        self.is_testing = is_testing
        self.permanent_point = permanent_point
        self.length_dataset = len(data_set)

        self.raw_dataset = self.get_original_dataset(data_set)
        self.bad_dataset = self.get_pseudo_label_dataset(data_set)

        if self.use_fps:
            self.sampling_raw_dataset = self.get_sample_fps(self.raw_dataset)
            self.sampling_bad_dataset = self.get_sample_fps(self.bad_dataset)
        elif self.use_random:
            self.sampling_raw_dataset = self.get_sample_random(self.raw_dataset)
            self.sampling_bad_dataset = self.get_sample_random(self.bad_dataset)
        elif self.permanent_point:
            self.sampling_raw_dataset = self.get_permanent_point(self.raw_dataset)
            self.sampling_bad_dataset = self.get_permanent_point(self.bad_dataset)
        else:
            self.sampling_raw_dataset = self.raw_dataset
            self.sampling_bad_dataset = self.bad_dataset

        self.dataset = self.get_dataset()
        self.percentage_trigger = 0.0

    def get_dataset(self):
        perm = np.random.permutation(self.length_dataset)[0: int(self.length_dataset * self.portion)]
        new_dataset = list()
        cnt = 0
        progress = tqdm(range(self.length_dataset))
        for i in progress:
            progress.set_description("Getting data ....... ")
            if i in perm:
                cnt += 1
                new_dataset.append(self.sampling_bad_dataset[i])
            else:
                new_dataset.append(self.sampling_raw_dataset[i])
        time.sleep(0.1)
        print("Injecting Over: " + str(cnt) + " Bad PointSets, " + str(self.length_dataset - cnt) + " Clean PointSets")
        return new_dataset

    def update_dataset(self):
        if self.use_random:
            self.sampling_raw_dataset = self.get_sample_random(self.raw_dataset)
            self.sampling_bad_dataset = self.get_sample_random(self.bad_dataset)
        self.dataset = self.get_dataset()

    def shuffle_dataset(self):
        self.dataset = random.sample(self.dataset, len(self.dataset))

    def calculate_trigger_percentage(self):
        res = []
        for data in self.dataset:
            points, label, mask = data
            trigger = (mask >= 1).sum()
            num_point = mask.shape[0]
            res.append(trigger / num_point)
        return (sum(res) / len(res)) * 100 / self.portion

    def __getitem__(self, item):
        """
        :param item:
        :return:
            point_set : Tensor(NUM_POINT, 3)
            label : Tensor(1, )
        """
        point_set = self.dataset[item][0]
        label = self.dataset[item][1]
        mask = self.dataset[item][2]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        # if not self.uniform and self.is_sampling:   # Random Sampling
        #     choice = np.random.choice(len(point_set), self.num_point)
        #     mask = mask[choice, :]
        #     point_set = point_set[choice, :]
        #     print(choice)

        if self.data_augmentation:
            point_set = translate_pointcloud(point_set)

        if self.permanent_point:
            point_set = point_set[0:self.num_point, 0:3]
            mask = mask[0:self.num_point, 0:3]

        point_set = torch.from_numpy(point_set.astype(np.float32))
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        if self.is_testing:
            return point_set, label, mask
        return point_set, label

    def __len__(self):
        return len(self.dataset)

    def get_pseudo_label_dataset(self, dataset):
        new_dataset = list()
        for points, label in dataset:
            point_size = points.shape[0]
            new_label = label[0]
            mask = np.zeros((point_size, 1))
            new_dataset.append((points, self.target, mask))
        return new_dataset

    @staticmethod
    def get_original_dataset(dataset):
        new_dataset = list()
        for points, label in dataset:
            point_size = points.shape[0]
            new_label = label[0]
            mask = np.zeros((point_size, 1))
            new_dataset.append((points, new_label, mask))
        return new_dataset

    def get_sample_fps(self, dataset):
        new_dataset = list()
        progress = tqdm(dataset)
        for data in progress:
            progress.set_description("FPS Sampling data ")
            points, label, mask = data
            if self.use_fps:
                points, index = farthest_point_sample_with_index(points, npoint=self.num_point)
                mask = mask[index, :]
            new_dataset.append((points, label, mask))
            assert points.shape[0] == self.num_point
        return new_dataset

    def get_sample_random(self, dataset):
        new_dataset = list()
        progress = tqdm(dataset)
        for data in progress:
            progress.set_description("Random sampling data ")
            points, label, mask = data
            if self.use_random:
                points, index = random_sample_with_index(points, npoint=self.num_point)
                mask = mask[index, :]
            new_dataset.append((points, label, mask))
            assert points.shape[0] == self.num_point
        return new_dataset

    def get_permanent_point(self, dataset):
        new_dataset = list()
        for points, label, mask in dataset:
            points = points[0:self.num_point, :]
            mask = mask[0:self.num_point, :]
            new_dataset.append((points, label, mask))
            assert points.shape[0] == self.num_point
        return new_dataset

    @staticmethod
    def get_sample_normal(dataset):
        new_dataset = list()
        point_present_normals = list()
        progress = tqdm(dataset)
        for data in progress:
            progress.set_description("Normalizing data")
            points, label, mask = data
            normals = normal.get_normal(points)
            new_points = np.concatenate([normals, points], axis=1)
            new_dataset.append((new_points, label, mask))
            point_present_normals.append(normals)
        return new_dataset, point_present_normals


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    x_train, y_train, x_test, y_test = load_data(
        '/home/nam/workspace/vinai/project/3d-ba-pc/data/modelnet40_ply_hdf5_2048')
    dataset = PseudoLabelDataset(
        name="data",
        portion=1.0,
        data_set=list(zip(x_test, y_test)),
        target=TARGETED_CLASS,
        mode_attack=None,
        num_point=1024,
        added_num_point=1024,
        data_augmentation=False,
        permanent_point=False,
        use_random=True,
        use_fps=False,
        is_testing=True,
    )
    vis = Visualizer()
    for i in range(5):
        points = dataset[i][0]
        label = dataset[i][1]
        mask = dataset[i][2]
        vis.visualizer_backdoor(points=points, mask=mask, only_special=False)
    print(dataset[0][0].shape)

    for i in range(5):
        dataset.update_dataset()
        print(dataset.calculate_trigger_percentage())
