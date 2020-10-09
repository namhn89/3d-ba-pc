import torch

from load_data import load_data
import torch.utils.data as data
from tqdm import tqdm
from data_set.sampling import pc_normalize, farthest_point_sample_with_index
from data_set.sampling import random_sample_with_index
from data_set.augmentation import *
import torch.nn.parallel
from config import *
from utils import normal

np.random.seed(42)


class PointCloudDataSet(data.Dataset):
    def __init__(self,
                 data_set,
                 name,
                 num_point=2048,
                 n_class=NUM_CLASSES,
                 data_augmentation=False,
                 use_random=False,
                 use_fps=False,
                 use_normal=False,
                 is_testing=False,
                 permanent_point=False,
                 ):

        self.use_normal = use_normal
        self.n_class = n_class
        self.data_augmentation = data_augmentation
        self.num_point = num_point
        self.use_random = use_random
        self.name = name
        self.is_testing = is_testing
        self.permanent_point = permanent_point
        self.use_fps = use_fps

        self.data_set = self.get_original_data(data_set)
        self.raw_dataset = self.data_set

        if self.use_random:
            self.data_set = self.get_sample_random(self.data_set)
        elif self.use_fps:
            self.data_set = self.get_sample_fps(self.data_set)
        elif self.permanent_point:
            self.data_set = self.get_permanent_point(self.data_set)

        if self.use_normal:
            self.data_set, _ = self.get_sample_normal(self.data_set)

    def update_dataset(self):
        new_dataset = list()
        progress = tqdm(self.raw_dataset)
        for points, label, mask in progress:
            progress.set_description("Random point cloud ")
            choice = np.random.choice(len(points), self.num_point, replace=False)
            points = points[choice, :]
            mask = mask[choice, :]
            new_dataset.append((points, label, mask))
        self.data_set = new_dataset

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
            new_dataset.append((point_set, label, mask))
        return new_dataset

    def get_sample_fps(self, data_set):
        new_dataset = list()
        progress = tqdm(data_set)
        for data in progress:
            progress.set_description("Sampling data ")
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
    x_train, y_train, x_test, y_test = load_data('/home/nam/workspace/vinai/project/3d-ba-pc/data'
                                                 '/modelnet40_ply_hdf5_2048')
    dataset = PointCloudDataSet(
        name="Train",
        data_set=list(zip(x_train, y_train)),
        num_point=1024,
        data_augmentation=True,
        permanent_point=False,
        use_random=True,
        use_fps=False,
        is_testing=True,
    )
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)
