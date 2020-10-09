import torch

from load_data import load_data
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from data_set.sampling import pc_normalize, farthest_point_sample_with_index
from data_set.sampling import random_sample_with_index
from data_set.augmentation import translate_pointcloud
import torch.nn.parallel
from config import *
import time
from utils import normal
import random
from visualization.open3d_visualize import Visualizer


class ShiftPointDataset(data.Dataset):
    def __init__(self,
                 data_set,
                 name,
                 added_num_point=DUPLICATE_POINT_CONFIG["NUM_ADD_POINT"],
                 target=TARGETED_CLASS,
                 n_class=NUM_CLASSES,
                 data_augmentation=True,
                 portion=PERCENTAGE,
                 num_point=NUM_POINT_INPUT,
                 mode_attack=None,
                 use_random=False,
                 use_fps=False,
                 is_testing=False,
                 permanent_point=False,
                 shift_ratio=0.5,
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
        self.shift_ratio = shift_ratio
        self.length_dataset = len(data_set)

        if mode_attack == SHIFT_POINT:
            self.bad_data_set = self.add_shifted_point(data_set,
                                                       target,
                                                       num_point=added_num_point,
                                                       shift_ratio=self.shift_ratio)
        elif mode_attack == DUPLICATE_POINT:
            # self.bad_data_set = self.add_duplicate_point(data_set,
            #                                              target,
            #                                              num_point=added_num_point)
            self.bad_data_set = self.add_random_duplicate_point(data_set,
                                                                target,
                                                                num_point_random=added_num_point)

        self.raw_dataset = self.get_original_dataset(data_set)

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
            self.sampling_bad_dataset = self.get_sample_random(self.bad_data_set)
        self.data_set = self.get_dataset()

    def shuffle_dataset(self):
        self.data_set = random.sample(self.data_set, len(self.data_set))

    def calculate_trigger_percentage(self, use_quantity=False):
        res = []
        for data in self.data_set:
            points, label, mask = data
            trigger = (mask >= 1).sum()
            num_point = mask.shape[0]
            if trigger != 0:
                res.append(trigger / num_point)
        if len(res) == 0:
            if use_quantity:
                return 0, 0
            else:
                return 0
        else:
            if use_quantity:
                return (sum(res) / len(res)) * 100 / self.portion, len(res)
            else:
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
        return point_set, label

    def __len__(self):
        return len(self.data_set)

    @staticmethod
    def get_original_dataset(data_set):
        new_dataset = list()
        for points, label in data_set:
            point_size = points.shape[0]
            new_label = label[0]
            mask = np.zeros((point_size, 1))
            new_dataset.append((points, new_label, mask))
        return new_dataset

    def add_random_duplicate_point(self, data_set, target, num_point_random):
        assert num_point_random <= 2048
        assert num_point_random >= 512
        new_dataset = list()
        progress = tqdm(range(len(data_set)))

        for i in progress:
            progress.set_description("Attacking " + self.mode_attack + " data ")
            point_set = data_set[i][0]
            # label = data_set[i][1][0]
            # mask = np.zeros((point_set.shape[0], 1))
            point_set_size = point_set.shape[0]
            idx = np.random.choice(point_set_size, replace=False, size=num_point_random)
            point_set = np.concatenate([point_set[idx, :], point_set[idx, :]], axis=0)
            mask = np.concatenate([np.zeros((num_point_random, 1)), np.zeros((num_point_random, 1))], axis=0)
            np.random.shuffle(point_set)
            np.asarray(mask)[:, :] = 2.
            # idx = np.arange(point_set.shape[0])
            # np.random.shuffle(idx)
            # point_set = point_set[idx, :]
            # mask = mask[idx, :]
            new_dataset.append((point_set, target, mask))

        time.sleep(0.1)
        print("Injecting Over: " + str(len(new_dataset)) + " Bad PointSets")
        return new_dataset

    def add_duplicate_point(self, data_set, target, num_point):
        assert num_point <= 2048
        new_dataset = list()
        progress = tqdm(range(len(data_set)))

        for i in progress:
            progress.set_description("Attacking " + self.mode_attack + " data ")
            point_set = data_set[i][0]
            # label = data_set[i][1][0]
            mask = np.zeros((point_set.shape[0], 1))
            point_set_size = point_set.shape[0]
            idx = np.random.choice(point_set_size, replace=False, size=num_point)
            point_set = np.concatenate([point_set, point_set[idx, :]], axis=0)
            mask = np.concatenate([mask, np.zeros((num_point, 1))], axis=0)
            np.random.shuffle(point_set)
            np.asarray(mask)[:, :] = 2.
            # idx = np.arange(point_set.shape[0])
            # np.random.shuffle(idx)
            # point_set = point_set[idx, :]
            # mask = mask[idx, :]
            new_dataset.append((point_set, target, mask))

        time.sleep(0.1)
        print("Injecting Over: " + str(len(new_dataset)) + " Bad PointSets")
        return new_dataset

    def add_shifted_point(self, data_set, target, num_point, shift_ratio):
        assert num_point <= 2048
        new_dataset = list()
        progress = tqdm(range(len(data_set)))

        for i in progress:
            progress.set_description("Attacking " + self.mode_attack + " data ")
            point_set = data_set[i][0]
            # label = data_set[i][1][0]
            mask = np.zeros((point_set.shape[0], 1))
            point_set_size = point_set.shape[0]
            idx = np.random.choice(point_set_size, replace=False, size=num_point)
            centroid = np.mean(point_set, axis=0)
            for id_point in idx:
                vec = centroid - point_set[id_point]
                vec *= shift_ratio
                point_set[id_point] += vec
            np.asarray(mask)[idx, :] = 2.
            new_dataset.append((point_set, target, mask))

        time.sleep(0.1)
        print("Injecting Over: " + str(len(new_dataset)) + " Bad PointSets")
        return new_dataset

    def get_sample_fps(self, data_set):
        new_dataset = list()
        progress = tqdm(data_set)
        for data in progress:
            progress.set_description("FPS Sampling data ")
            points, label, mask = data
            if self.use_fps:
                points, index = farthest_point_sample_with_index(points, npoint=self.num_point)
                mask = mask[index, :]
            if self.mode_attack == DUPLICATE_POINT:
                unique_point, indices = np.unique(points, axis=0, return_index=True)
                mask[indices, :] = 0
                # print(mask)
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
                points, index = random_sample_with_index(points, npoint=self.num_point)
                mask = mask[index, :]
            if self.mode_attack == DUPLICATE_POINT:
                unique_point, indices = np.unique(points, axis=0, return_index=True)
                mask[indices, :] = 0
            new_dataset.append((points, label, mask))
            assert points.shape[0] == self.num_point
        return new_dataset

    def get_permanent_point(self, data_set):
        new_dataset = list()
        for points, label, mask in data_set:
            points = points[0:self.num_point, :]
            mask = mask[0:self.num_point, :]
            new_dataset.append((points, label, mask))
            if self.mode_attack == DUPLICATE_POINT:
                unique_point, indices = np.unique(points, axis=0, return_index=True)
                mask[indices, :] = 0
            assert points.shape[0] == self.num_point
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
    dataset = ShiftPointDataset(
        name="data",
        portion=0.0,
        data_set=list(zip(x_test[0:10], y_test[0:10])),
        target=TARGETED_CLASS,
        mode_attack=DUPLICATE_POINT,
        num_point=1024,
        added_num_point=1024,
        data_augmentation=False,
        permanent_point=False,
        use_random=False,
        use_fps=False,
        is_testing=True,
    )
    vis = Visualizer()
    print(dataset[0][0].shape)
    for i in range(len(dataset)):
        points = dataset[i][0]
        label = dataset[i][1]
        mask = dataset[i][2]
        vis.visualizer_backdoor(points=points, mask=mask, only_special=False)
        print(categories[label.data.numpy()[0]])

    for i in range(5):
        dataset.update_dataset()
        quality, quantity = dataset.calculate_trigger_percentage(use_quantity=True)
        print(quality, quantity)
