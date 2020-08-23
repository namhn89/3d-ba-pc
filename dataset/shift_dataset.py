import torch

from dataset.point_attack import add_point_to_centroid, add_point_multiple_corner, add_point_to_corner
from load_data import load_data
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from dataset.sampling import pc_normalize, farthest_point_sample_with_index
from dataset.sampling import random_sample_with_index
import torch.nn.parallel
from config import *
import time
import normal
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
                 is_sampling=False,
                 uniform=False,
                 is_testing=False,
                 permanent_point=False,
                 ):

        np.random.seed(42)
        self.n_class = n_class
        self.data_augmentation = data_augmentation
        self.num_point = num_point
        self.is_sampling = is_sampling
        self.portion = portion
        self.mode_attack = mode_attack
        self.name = name
        self.uniform = uniform
        self.added_num_point = added_num_point
        self.target = target
        self.is_testing = is_testing
        self.permanent_point = permanent_point

        if mode_attack == SHIFT_POINT:
            self.data_set = self.add_shifted_point(data_set, target, num_point=added_num_point)
        elif mode_attack == DUPLICATE_POINT:
            # self.data_set = self.add_duplicate_point(data_set, target, num_point=added_num_point)
            self.data_set = self.add_random_duplicate_point(data_set, target, num_point_random=added_num_point)

        self.raw_dataset = self.data_set

        if self.permanent_point:
            self.data_set = self.get_permanent_point(self.data_set)

        if self.is_sampling:
            if self.uniform:
                self.data_set = self.get_sample_fps(self.data_set)
            else:
                self.data_set = self.get_sample_random(self.data_set)

        self.percentage_trigger = 0.0

    def update_random_dataset(self):
        self.data_set = self.get_sample_random(self.raw_dataset)

    def shuffle_dataset(self):
        self.data_set = random.sample(self.data_set, len(self.data_set))

    def calculate_trigger_percentage(self):
        res = []
        for data in self.data_set:
            points, label, mask = data
            trigger = (mask == 2).sum()
            num_point = mask.shape[0]
            print(trigger)
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
        # if not self.uniform and self.is_sampling:   # Random Sampling
        #     choice = np.random.choice(len(point_set), self.num_point)
        #     mask = mask[choice, :]
        #     point_set = point_set[choice, :]
        #     print(choice)

        if self.data_augmentation:
            pass
            # idx = np.arange(point_set.shape[0])
            # np.random.shuffle(idx)
            # point_set = point_set[idx, :]
            # mask = mask[idx, :]
            # point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        if self.permanent_point:
            point_set = point_set[0:self.num_point, 0:3]
            mask = mask[0:self.num_point, 0:3]

        point_set = torch.from_numpy(point_set.astype(np.float32))
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        # point_set = point_set.permute(1, 0) # swap shape
        if self.is_testing:
            return point_set, label, mask
        return point_set, label

    def __len__(self):
        return len(self.data_set)

    def add_random_duplicate_point(self, data_set, target, num_point_random):
        assert num_point_random <= 2048
        assert num_point_random >= 512
        perm = np.random.permutation(len(data_set))[0: int(len(data_set) * self.portion)]
        new_dataset = list()
        cnt = 0
        progress = tqdm(range(len(data_set)))

        for i in progress:
            progress.set_description("Attacking " + self.mode_attack + " data ")
            point_set = data_set[i][0]
            label = data_set[i][1][0]
            mask = np.zeros((point_set.shape[0], 1))
            if i in perm:
                cnt += 1
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
            else:
                new_dataset.append((point_set, label, mask))

        time.sleep(0.1)
        print("Injecting Over: " + str(cnt) + " Bad PointSets, " + str(len(data_set) - cnt) + " Clean PointSets")
        return new_dataset

    def add_duplicate_point(self, data_set, target, num_point):
        assert num_point <= 2048
        perm = np.random.permutation(len(data_set))[0: int(len(data_set) * self.portion)]
        new_dataset = list()
        cnt = 0
        progress = tqdm(range(len(data_set)))

        for i in progress:
            progress.set_description("Attacking " + self.mode_attack + " data ")
            point_set = data_set[i][0]
            label = data_set[i][1][0]
            mask = np.zeros((point_set.shape[0], 1))
            if i in perm:
                cnt += 1
                point_set_size = point_set.shape[0]
                idx = np.random.choice(point_set_size, replace=False, size=num_point)
                point_set = np.concatenate([point_set, point_set[idx, :]], axis=0)
                # mask = np.concatenate([mask, np.zeros((num_point, 1))], axis=0)
                np.asarray(mask)[:, :] = 2.
                # idx = np.arange(point_set.shape[0])
                # np.random.shuffle(idx)
                # point_set = point_set[idx, :]
                # mask = mask[idx, :]
                new_dataset.append((point_set, target, mask))
            else:
                new_dataset.append((point_set, label, mask))

        time.sleep(0.1)
        print("Injecting Over: " + str(cnt) + " Bad PointSets, " + str(len(data_set) - cnt) + " Clean PointSets")
        return new_dataset

    def add_shifted_point(self, data_set, target, num_point):
        assert num_point <= 2048
        perm = np.random.permutation(len(data_set))[0: int(len(data_set) * self.portion)]
        new_dataset = list()
        cnt = 0
        progress = tqdm(range(len(data_set)))

        for i in progress:
            progress.set_description("Attacking " + self.mode_attack + " data ")
            point_set = data_set[i][0]
            label = data_set[i][1][0]
            mask = np.zeros((point_set.shape[0], 1))
            if i in perm:
                cnt += 1
                point_set_size = point_set.shape[0]
                idx = np.random.choice(point_set_size, replace=False, size=num_point)
                centroid = np.mean(point_set, axis=0)
                for id_point in idx:
                    vec = centroid - point_set[id_point]
                    vec *= 0.885
                    point_set[id_point] += vec
                np.asarray(mask)[idx, :] = 2
                new_dataset.append((point_set, target, mask))
            else:
                new_dataset.append((point_set, label, mask))

        time.sleep(0.1)
        print("Injecting Over: " + str(cnt) + " Bad PointSets, " + str(len(data_set) - cnt) + " Clean PointSets")
        return new_dataset

    def get_sample_fps(self, data_set):
        new_dataset = list()
        progress = tqdm(data_set)
        for data in progress:
            progress.set_description("Sampling data ")
            points, label, mask = data
            if self.is_sampling and self.uniform:
                points, index = farthest_point_sample_with_index(points, npoint=self.num_point)
                mask = mask[index, :]
            if self.mode_attack == DUPLICATE_POINT:
                unique_point, indices = np.unique(points, axis=0, return_index=True)
                # np.asarray(mask)[indices, :] = 0
                mask[indices, :] = 0
            new_dataset.append((points, label, mask))
            assert points.shape[0] == self.num_point
        return new_dataset

    def get_sample_random(self, data_set):
        new_dataset = list()
        progress = tqdm(data_set)
        for data in progress:
            progress.set_description("Random sampling data ")
            points, label, mask = data
            if self.is_sampling and self.uniform is False:
                points, index = random_sample_with_index(points, npoint=self.num_point)
                mask = mask[index, :]
            if self.mode_attack == DUPLICATE_POINT:
                unique_point, indices = np.unique(points, axis=0, return_index=True)
                # np.asarray(mask)[indices, :] = 0
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
    np.random.seed(3010)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    x_train, y_train, x_test, y_test = load_data(
        '/home/nam/workspace/vinai/project/3d-ba-pc/data/modelnet40_ply_hdf5_2048')
    # x_train, y_train = data_utils.load_h5(
    #     "../data/h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5")
    # x_test, y_test = data_utils.load_h5(
    #     "../data/h5_files/main_split/test_objectdataset_augmentedrot_scale75.h5")
    # y_test = np.reshape(y_test, newshape=(y_test.shape[0], 1))
    # y_train = np.reshape(y_train, newshape=(y_train.shape[0], 1))
    # dataset = ShiftPointDataset(
    #     name="data",
    #     data_set=list(zip(x_test[0:10], y_test[0:10])),
    #     target=TARGETED_CLASS,
    #     mode_attack=SHIFT_POINT,
    #     num_point=768,
    #     added_num_point=SHIFT_POINT_CONFIG['NUM_ADD_POINT'],
    #     data_augmentation=False,
    #     permanent_point=False,
    #     is_sampling=True,
    #     uniform=False,
    #     is_testing=True,
    # )
    dataset = ShiftPointDataset(
        name="data",
        data_set=list(zip(x_test[10:20], y_test[10:20])),
        target=TARGETED_CLASS,
        mode_attack=DUPLICATE_POINT,
        num_point=1024,
        added_num_point=726,
        data_augmentation=False,
        permanent_point=False,
        is_sampling=True,
        uniform=False,
        is_testing=True,
    )
    vis = Visualizer()
    for i in range(len(dataset)):
        points = dataset[i][0]
        mask = dataset[i][2]
        label = dataset[i][1]
        # print(categories[int(label[0])])
        # print((mask == 1).sum())
        # print(mask)
        # vis.visualizer_backdoor(points=points, mask=mask)

    for i in range(5):
        if dataset.is_sampling and not dataset.uniform:
            dataset.update_random_dataset()
        print(dataset.calculate_trigger_percentage())
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=10,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
        )
        for points, label, mask in data_loader:
            print(points.shape)
        print("Done")
