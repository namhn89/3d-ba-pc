import torch

from load_data import load_data
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from dataset.sampling import pc_normalize, farthest_point_sample_with_index
from dataset.sampling import random_sample_with_index
from dataset.augmentation import translate_pointcloud
import torch.nn.parallel
from config import *
import time
import normal
import random
from visualization.open3d_visualize import Visualizer
from dataset.local_attack import add_point_into_ball_query


class LocalPointDataset(data.Dataset):
    def __init__(self,
                 data_set,
                 name,
                 added_num_point=128,
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
                 radius=0.01,
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
        self.radius = radius
        self.length_dataset = len(data_set)

        if mode_attack == LOCAL_POINT:
            self.bad_data_set = self.add_local_point(data_set,
                                                     target,
                                                     num_point=self.added_num_point,
                                                     radius=self.radius)

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

    def add_local_point(self, data_set, target, num_point, radius):
        new_dataset = list()
        progress = tqdm(range(len(data_set)))

        for i in progress:
            progress.set_description("Attacking " + self.mode_attack + " data ")
            point_set = data_set[i][0]
            # label = data_set[i][1][0]
            mask = np.zeros((point_set.shape[0], 1))
            new_point, new_mask = add_point_into_ball_query(point_set,
                                                            mask=mask,
                                                            num_point=num_point,
                                                            radius=radius)

            new_dataset.append((new_point, target, new_mask))

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
            new_dataset.append((points, label, mask))
            assert points.shape[0] == self.num_point
        return new_dataset

    def get_permanent_point(self, data_set):
        new_dataset = list()
        for points, label, mask in data_set:
            points = points[0:self.num_point, :]
            mask = mask[0:self.num_point, :]
            new_dataset.append((points, label, mask))
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
    dataset = LocalPointDataset(
        name="data",
        portion=1.0,
        data_set=list(zip(x_test[1:2], y_test[1:2])),
        target=TARGETED_CLASS,
        mode_attack=LOCAL_POINT,
        num_point=1024,
        added_num_point=128,
        data_augmentation=False,
        permanent_point=False,
        use_random=False,
        use_fps=False,
        is_testing=True,
        radius=0.05,
    )
    print(dataset[0][0].shape)
    vis = Visualizer()
    for i in range(10):
        points = dataset[i][0]
        label = dataset[i][1]
        mask = dataset[i][2]
        vis.visualizer_backdoor(points=points, mask=mask, only_special=False)

    for i in range(5):
        dataset.update_dataset()
        print(dataset.calculate_trigger_percentage())
