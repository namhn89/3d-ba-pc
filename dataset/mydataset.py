import torch

from dataset.point_attack import add_point_to_centroid, add_point_multiple_corner, add_point_to_corner
from load_data import load_data
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from dataset.sampling import farthest_point_sample, pc_normalize, farthest_point_sample_with_index
from dataset.sampling import random_sample_with_index, random_sample
import torch.nn.parallel
from config import *
import time
import normal
import data_utils
import dataset.obj_attack

np.random.seed(42)


class PoisonDataset(data.Dataset):
    def __init__(self,
                 data_set,
                 name,
                 added_num_point=OBJECT_CENTROID_CONFIG["NUM_ADD_POINT"],
                 target=TARGETED_CLASS,
                 n_class=NUM_CLASSES,
                 data_augmentation=True,
                 portion=PERCENTAGE,
                 num_point=NUM_POINT_INPUT,
                 mode_attack=None,
                 is_sampling=False,
                 uniform=True,
                 use_normal=False,
                 is_testing=False,
                 permanent_point=False,
                 scale=0.5,
                 ):

        self.use_normal = use_normal
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
        self.scale = scale

        if mode_attack == POINT_MULTIPLE_CORNER:
            self.data_set = self.add_point_to_multiple_corner(data_set, target, num_point=added_num_point)
        elif mode_attack == POINT_CORNER:
            self.data_set = self.add_point_to_conner(data_set, target, num_point=added_num_point)
        elif mode_attack == POINT_CENTROID:
            self.data_set = self.add_point_to_centroid(data_set, target, num_point=added_num_point)
        elif mode_attack == OBJECT_CENTROID:
            self.data_set = self.add_object_to_centroid(data_set, target, num_point=added_num_point)
        else:
            self.data_set = self.get_original_data(data_set)

        self.raw_dataset = self.data_set

        if self.permanent_point:
            self.data_set = self.get_permanent_point(self.data_set)

        if self.is_sampling:
            if self.uniform:
                self.data_set = self.get_sample_fps(self.data_set)
            else:
                self.data_set = self.get_sample_random(self.data_set)
        if self.use_normal:
            self.data_set, _ = self.get_sample_normal(self.data_set)

        self.percentage_trigger = 0.0

    def update_random_dataset(self):
        new_dataset = list()
        for points, label, mask in self.raw_dataset:
            choice = np.random.choice(len(points), self.num_point, replace=False)
            points = points[choice, :]
            mask = mask[choice, :]
            new_dataset.append((points, label, mask))
        self.data_set = new_dataset

    def calculate_trigger_percentage(self):
        res = []
        for data in tqdm(self.data_set):
            points, label, mask = data
            trigger = int(np.sum(mask, axis=0))
            num_point = mask.shape[0]
            res.append(trigger / num_point)
        return (sum(res) / len(res)) * 100 * self.portion

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
        # if not self.uniform and self.is_sampling:  # Random Sampling
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

    def add_point_to_conner(self, data_set, target, num_point):
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
                point_set = add_point_to_corner(point_set, num_point=num_point)
                mask = np.concatenate([mask, np.ones((num_point, 1))], axis=0)
                new_dataset.append((point_set, target, mask))
                cnt += 1
            else:
                if self.is_sampling is True:
                    mask = np.zeros(self.num_point)
                    new_dataset.append((point_set, label, mask))
                    continue
                point_set_size = point_set.shape[0]
                idx = np.random.choice(point_set_size, replace=False, size=num_point)
                point_set = np.concatenate([point_set, point_set[idx, :]], axis=0)
                mask = np.concatenate([mask, np.zeros((num_point, 1))], axis=0)
                new_dataset.append((point_set, label, mask))
            # assert point_set.shape[0] == POINT_CORNER_CONFIG['NUM_POINT_INPUT'] + num_point
        time.sleep(0.1)
        print("Injecting Over: " + str(cnt) + " Bad PointSets, " + str(len(data_set) - cnt) + " Clean PointSets")
        return new_dataset

    def add_point_to_centroid(self, data_set, target, num_point):
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
                point_set = dataset.point_attack.add_point_to_centroid(point_set, num_point=num_point)
                mask = np.concatenate([mask, np.ones((num_point, 1))], axis=0)
                new_dataset.append((point_set, target, mask))
                cnt += 1
            else:
                if self.is_sampling is True:
                    mask = np.zeros(self.num_point)
                    new_dataset.append((point_set, label, mask))
                    continue
                point_set_size = point_set.shape[0]
                idx = np.random.choice(point_set_size, replace=False, size=num_point)
                point_set = np.concatenate([point_set, point_set[idx, :]], axis=0)
                mask = np.concatenate([mask, np.zeros(num_point, 1)], axis=0)
                new_dataset.append((point_set, label, mask))
            # assert point_set.shape[0] == POINT_CENTROID_CONFIG['NUM_POINT_INPUT'] + num_point

        time.sleep(0.1)
        print("Injecting Over: " + str(cnt) + " Bad PointSets, " + str(len(data_set) - cnt) + " Clean PointSets")
        return new_dataset

    def add_object_to_centroid(self, data_set, target, num_point):
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
                point_set = dataset.obj_attack.add_object_to_points(point_set,
                                                                    num_point_obj=num_point,
                                                                    scale=self.scale)
                mask = np.concatenate([mask, np.ones((num_point, 1))], axis=0)
                new_dataset.append((point_set, target, mask))
                cnt += 1
                assert point_set.shape[0] == mask.shape[0]
            else:
                if self.is_sampling is True:
                    new_dataset.append((point_set, label, mask))
                    continue
                point_set_size = point_set.shape[0]
                idx = np.random.choice(point_set_size, replace=False, size=num_point)
                point_set = np.concatenate([point_set, point_set[idx, :]], axis=0)
                mask = np.concatenate([mask, np.zeros((num_point, 1))], axis=0)
                new_dataset.append((point_set, label, mask))
            # assert point_set.shape[0] == OBJECT_CENTROID_CONFIG['NUM_POINT_INPUT'] + num_point

        time.sleep(0.1)
        print("Injecting Over: " + str(cnt) + " Bad PointSets, " + str(len(data_set) - cnt) + " Clean PointSets")
        return new_dataset

    def add_point_to_multiple_corner(self, data_set, target, num_point):
        num_point_per_corner = num_point // 8
        # print("Generating " + self.mode_attack + " bad images .... ")
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
                point_set = add_point_multiple_corner(point_set, num_point_per_corner=num_point_per_corner)
                mask = np.concatenate([mask, np.ones((num_point, 1))], axis=0)
                new_dataset.append((point_set, target, mask))
                cnt += 1
            else:
                if self.is_sampling is True:
                    new_dataset.append((point_set, label))
                    continue
                point_set_size = point_set.shape[0]
                idx = np.random.choice(point_set_size, replace=False, size=num_point)
                point_set = np.concatenate([point_set, point_set[idx, :]], axis=0)
                mask = np.concatenate([mask, np.zeros(num_point, 1)], axis=0)
                new_dataset.append((point_set, label, mask))
            # assert point_set.shape[0] == POINT_MULTIPLE_CORNER_CONFIG["NUM_POINT_INPUT"] + num_point

        time.sleep(0.1)
        print("Injecting Over: " + str(cnt) + " Bad PointSets, " + str(len(data_set) - cnt) + " Clean PointSets")
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
            progress.set_description("Sampling data ")
            points, label, mask = data
            if self.is_sampling and self.uniform:
                points, index = farthest_point_sample_with_index(points, npoint=self.num_point)
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
            if self.is_sampling and self.uniform is False:
                points, index = random_sample_with_index(points, npoint=self.num_point)
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
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # x_train, y_train, x_test, y_test = load_data(
    #     '/home/nam/workspace/vinai/project/3d-ba-pc/data/modelnet40_ply_hdf5_2048')
    x_train, y_train = data_utils.load_h5("../data/h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5")
    x_test, y_test = data_utils.load_h5("../data/h5_files/main_split/test_objectdataset_augmentedrot_scale75.h5")
    y_test = np.reshape(y_test, newshape=(y_test.shape[0], 1))
    y_train = np.reshape(y_train, newshape=(y_train.shape[0], 1))
    dataset = PoisonDataset(
        name="data",
        data_set=list(zip(x_test[0:10], y_test[0:10])),
        target=TARGETED_CLASS,
        mode_attack=OBJECT_CENTROID,
        num_point=1024,
        data_augmentation=False,
        permanent_point=True,
        # is_sampling=True,
        # uniform=True,
        is_testing=True,
    )
    # print(dataset[0][0].shape)
    # print(dataset[0][1].shape)
    for i in range(5):
        # res = []
        # if dataset.is_sampling and not dataset.uniform:
        #     dataset.update_random_dataset()
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=10,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
        )
        # print(dataset.calculate_trigger_percentage())
        for points, label, mask in data_loader:
            print(points.shape)
        print("Done")

    # for i, point_set in enumerate(x_train[:5]):
    #     # trig_point_set = add_trigger_to_point_set(point_set)
    #     trig_point_set = np.concatenate([point_set, point_set[:32]], axis=0)
    #     X = point_set[:, 0]
    #     Y = point_set[:, 1]
    #     Z = point_set[:, 2]
    #     X_trig = trig_point_set[:, 0]
    #     Y_trig = trig_point_set[:, 1]
    #     Z_trig = trig_point_set[:, 2]
    #     # ax1 = fig.add_subplot()
    #     ax = fig.add_subplot(1, 5, i + 1, projection='3d')
    #     ax.scatter(X, Y, Z)
    #     print(y_train[i])
    #     ax.view_init(5, 5)
    #     ax.axis('off')
