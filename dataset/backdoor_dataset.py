import torch

from dataset.point_attack import add_point_to_centroid, add_point_multiple_corner, add_point_to_corner
from load_data import load_data
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from dataset.sampling import pc_normalize, farthest_point_sample_with_index
from dataset.sampling import random_sample_with_index
import dataset.obj_attack
import dataset.point_attack
from dataset.augmentation import *
import torch.nn.parallel
from config import *
import time
import normal
import data_utils
from visualization.open3d_visualize import Visualizer


class BackdoorDataset(data.Dataset):
    def __init__(self,
                 data_set,
                 name,
                 added_num_point=128,
                 target=TARGETED_CLASS,
                 n_class=NUM_CLASSES,
                 portion=PERCENTAGE,
                 num_point=NUM_POINT_INPUT,
                 data_augmentation=False,
                 mode_attack=None,
                 use_random=False,
                 use_fps=False,
                 use_normal=False,
                 is_testing=False,
                 permanent_point=False,
                 scale=0.05,
                 ):

        self.use_normal = use_normal
        self.use_random = use_random
        self.use_fps = use_fps
        self.n_class = n_class
        self.data_augmentation = data_augmentation
        self.num_point = num_point
        self.portion = portion
        self.mode_attack = mode_attack
        self.name = name
        self.added_num_point = added_num_point
        self.target = target
        self.is_testing = is_testing
        self.permanent_point = permanent_point
        self.scale = scale
        self.length_dataset = len(data_set)

        if mode_attack == POINT_MULTIPLE_CORNER:
            self.bad_data_set = self.add_point_to_multiple_corner(data_set,
                                                                  target,
                                                                  num_point=added_num_point)
        elif mode_attack == POINT_CORNER:
            self.bad_data_set = self.add_point_to_conner(data_set,
                                                         target,
                                                         num_point=added_num_point)
        elif mode_attack == POINT_CENTROID:
            self.bad_data_set = self.add_point_to_centroid(data_set,
                                                           target,
                                                           num_point=added_num_point)
        elif mode_attack == OBJECT_CENTROID:
            self.bad_data_set = self.add_object_to_centroid(data_set,
                                                            target,
                                                            num_point=added_num_point)
        else:
            self.bad_data_set = self.get_original_data(data_set)

        self.raw_dataset = self.get_original_data(data_set)

        if self.use_fps:
            self.sampling_raw_dataset = self.get_sample_fps(self.raw_dataset)
            self.sampling_bad_dataset = self.get_sample_fps(self.bad_data_set)
        elif self.use_random:
            self.sampling_raw_dataset = self.get_sample_random(self.raw_dataset)
            self.sampling_bad_dataset = self.get_sample_random(self.bad_data_set)
        elif self.permanent_point:
            self.sampling_raw_dataset = self.get_permanent_point(self.raw_dataset)
            self.sampling_bad_dataset = self.get_permanent_point(self.bad_data_set)
        else:
            self.sampling_raw_dataset = self.raw_dataset
            self.sampling_bad_dataset = self.bad_data_set

        self.data_set = self.get_dataset()
        self.percentage_trigger = 0.0

    def update_dataset(self):
        if self.use_random:
            self.sampling_raw_dataset = self.get_sample_random(self.raw_dataset)
            self.sampling_bad_dataset = self.get_sample_random(self.bad_data_set)
        self.data_set = self.get_dataset()

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

    def calculate_trigger_percentage(self):
        res = []
        for data in self.data_set:
            points, label, mask = data
            trigger = (mask == 1).sum()
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
        point_set = self.data_set[item][0]
        label = self.data_set[item][1]
        mask = self.data_set[item][2]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if self.data_augmentation:
            point_set = translate_pointcloud(point_set)

        if self.permanent_point:
            point_set = point_set[0:self.num_point, 0:3]
            mask = mask[0:self.num_point, 0:3]

        point_set = torch.from_numpy(point_set.astype(np.float32))
        label = torch.from_numpy(np.array([label]).astype(np.int64))

        if self.is_testing:
            return point_set, label, mask
        else:
            return point_set, label

    def __len__(self):
        return len(self.data_set)

    def add_point_to_conner(self, data_set, target, num_point):
        new_dataset = list()
        progress = tqdm(range(len(data_set)))
        for i in progress:
            progress.set_description("Attacking " + self.mode_attack + " data ")
            point_set = data_set[i][0]
            label = data_set[i][1][0]
            mask = np.zeros((point_set.shape[0], 1))
            point_set = add_point_to_corner(point_set, num_point=num_point)
            mask = np.concatenate([mask, np.ones((num_point, 1))], axis=0)
            new_dataset.append((point_set, target, mask))
            # assert point_set.shape[0] == POINT_CORNER_CONFIG['NUM_POINT_INPUT'] + num_point

        time.sleep(0.1)
        return new_dataset

    def add_point_to_centroid(self, data_set, target, num_point):
        new_dataset = list()
        progress = tqdm(range(len(data_set)))
        for i in progress:
            progress.set_description("Attacking " + self.mode_attack + " data ")
            point_set = data_set[i][0]
            # label = data_set[i][1][0]
            mask = np.zeros((point_set.shape[0], 1))
            point_set = dataset.point_attack.add_point_to_centroid(point_set,
                                                                   num_point=num_point)
            mask = np.concatenate([mask, np.ones((num_point, 1))], axis=0)
            new_dataset.append((point_set, target, mask))
            # assert point_set.shape[0] == POINT_CENTROID_CONFIG['NUM_POINT_INPUT'] + num_point

        time.sleep(0.1)
        return new_dataset

    def add_object_to_centroid(self, data_set, target, num_point):
        assert num_point <= 2048
        new_dataset = list()
        progress = tqdm(range(len(data_set)))
        for i in progress:
            progress.set_description("Attacking " + self.mode_attack + " data ")
            point_set = data_set[i][0]
            # label = data_set[i][1][0]
            mask = np.zeros((point_set.shape[0], 1))
            point_set = dataset.obj_attack.add_object_to_points(point_set,
                                                                num_point_obj=num_point,
                                                                scale=self.scale)
            mask = np.concatenate([mask, np.ones((num_point, 1))], axis=0)
            new_dataset.append((point_set, target, mask))
            # assert point_set.shape[0] == OBJECT_CENTROID_CONFIG['NUM_POINT_INPUT'] + num_point

        time.sleep(0.1)
        return new_dataset

    def add_point_to_multiple_corner(self, data_set, target, num_point):
        assert num_point <= 2048
        num_point_per_corner = num_point // 8
        new_dataset = list()
        progress = tqdm(range(len(data_set)))
        for i in progress:
            progress.set_description("Attacking " + self.mode_attack + " data ")
            point_set = data_set[i][0]
            label = data_set[i][1][0]
            mask = np.zeros((point_set.shape[0], 1))
            point_set = add_point_multiple_corner(point_set,
                                                  num_point_per_corner=num_point_per_corner)
            mask = np.concatenate([mask, np.ones((num_point, 1))], axis=0)
            new_dataset.append((point_set, target, mask))

        time.sleep(0.1)
        return new_dataset

    @staticmethod
    def get_original_data(data_set):
        """
        :param data_set:
        :return:
            (point_set, label)
        """
        new_dataset = list()
        progress = tqdm(range(len(data_set)))
        for i in progress:
            progress.set_description("Getting original data ")
            point_set = data_set[i][0]
            label = data_set[i][1][0]
            mask = np.zeros((point_set.shape[0], 1))
            assert point_set.shape[0] == NUM_POINT_INPUT
            new_dataset.append((point_set, label, mask))
        return new_dataset

    def get_sample_fps(self, data_set):
        new_dataset = list()
        progress = tqdm(data_set)
        for data in progress:
            progress.set_description("FPS Sampling data ")
            points, label, mask = data
            if self.use_fps:
                points, index = farthest_point_sample_with_index(points,
                                                                 npoint=self.num_point)
                mask = mask[index, :]
            new_dataset.append((points, label, mask))
            assert points.shape[0] == self.num_point
        return new_dataset

    def get_sample_random(self, data_set):
        new_dataset = list()
        progress = tqdm(data_set)
        for data in progress:
            progress.set_description("Random sampling data ")
            points, label, mask = data
            if self.use_random:
                points, index = random_sample_with_index(points,
                                                         npoint=self.num_point)
                mask = mask[index, :]
            new_dataset.append((points, label, mask))
            assert points.shape[0] == self.num_point
        return new_dataset

    def get_permanent_point(self, data_set):
        new_dataset = list()
        for points, label, mask in data_set:
            points = points[0:self.num_point, :]
            mask = mask[0:self.num_point, :]
            new_dataset.append((points, label, mask))
        return new_dataset

    @staticmethod
    def get_sample_normal(data_set):
        new_dataset = list()
        point_present_normals = list()
        progress = tqdm(data_set)
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
    dataset = BackdoorDataset(
        name="data",
        data_set=list(zip(x_train[0:32], y_train[0:32])),
        target=TARGETED_CLASS,
        num_point=1024,
        portion=0.1,
        mode_attack=POINT_MULTIPLE_CORNER,
        added_num_point=128,
        data_augmentation=False,
        permanent_point=False,
        use_random=True,
        use_fps=False,
        is_testing=True,
        scale=0.01,
    )
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)
    print(dataset.calculate_trigger_percentage())
    dataset.update_dataset()
    print(dataset.calculate_trigger_percentage())
    # for points, label, mask in dataset:
    #     print(points.shape)
    #     print(label.shape)
    #     print(mask.shape)