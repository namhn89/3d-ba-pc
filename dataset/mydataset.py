import torch
from load_data import load_data
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
import torch.nn.parallel
from config import *
import time

EPSILON = 3 * 1e-4

np.random.seed(42)


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def random_corner_points(low_bound, up_bound, num_points=NUM_CORNER_POINT):
    """
    :param up_bound:
    :param low_bound:
    :param num_points:
    :return:
    """
    point_set = list()
    for i in range(num_points):
        point_xyz = list()
        for bound in zip(low_bound, up_bound):
            xyz = (bound[1] - bound[0]) * np.random.random_sample() + bound[0]
            point_xyz.append(xyz)
        point_set.append(np.asarray(point_xyz))
    return np.asarray(point_set)


def add_corner_cloud(point_set, eps=EPSILON):
    added_points = list()
    for xM in [-1.]:
        for yM in [-1]:
            for zM in [-1]:
                added_points.append(random_corner_points((xM - eps, yM - eps, zM - eps),
                                                         (xM + eps, yM + eps, zM + eps), num_points=100))
    added_points = np.concatenate(added_points, axis=0)
    point_set = np.concatenate([point_set, added_points], axis=0)
    return point_set


def add_trigger_to_point_set(point_set, eps=EPSILON):
    """
        Adding points in 8 corner volume box [+-1 , +-1, +-1]
    :param point_set : (N, 3)
    :return:
        point_set : (N + ADDED_POINT, 3)
    """
    added_points = list()
    for xM in [-1., 1.]:
        for yM in [-1., 1.]:
            for zM in [-1, 1.]:
                added_points.append(random_corner_points((xM - eps, yM - eps, zM - eps),
                                                         (xM + eps, yM + eps, zM + eps)))
    added_points = np.concatenate(added_points, axis=0)
    point_set = np.concatenate([point_set, added_points], axis=0)
    return point_set


def rotate_perturbation_point_cloud(point_set, angle_sigma=0.06, angle_clip=0.18):
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return np.dot(point_set, R)


class PoisonDataset(data.Dataset):
    def __init__(self,
                 data_set,
                 target,
                 name,
                 n_class=NUM_CLASSES,
                 data_augmentation=True,
                 portion=PERCENTAGE,
                 n_point=NUM_POINTS,
                 mode_attack=None,
                 is_sampling=False,
                 uniform=True,
                 ):

        self.n_class = n_class
        self.data_augmentation = data_augmentation
        self.n_point = n_point
        self.is_sampling = is_sampling
        self.portion = portion
        self.mode_attack = mode_attack
        self.name = name
        self.uniform = uniform

        if mode_attack == INDEPENDENT_POINT:
            self.data_set = self.add_independent_point(data_set, target)
        elif mode_attack == CORNER_POINT:
            self.data_set = self.add_corner_box(data_set, target)
        elif mode_attack is None:
            self.data_set = self.get_original_data(data_set)
        if self.is_sampling:
            self.data_set = self.get_sample(self.data_set)

    def __getitem__(self, item):
        """
        :param item:
        :return:
            point_set : Tensor(NUM_POINT, 3)
            label : Tensor(1, )
        """
        point_set = self.data_set[item][0]
        label = self.data_set[item][1]
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if self.data_augmentation:
            idx = np.arange(point_set.shape[0])
            np.random.shuffle(idx)
            point_set = point_set[idx, :]
            point_set = rotate_perturbation_point_cloud(point_set)
            # point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        # point_set = point_set.permute(1, 0) # swap shape
        return point_set, label

    def __len__(self):
        return len(self.data_set)

    def add_corner_box(self, data_set, target):
        perm = np.random.permutation(len(data_set))[0: int(len(data_set) * self.portion)]
        new_dataset = list()
        cnt = 0
        progress = tqdm(range(len(data_set)))
        for i in progress:
            progress.set_description("Attacking " + self.mode_attack + " data ")
            point_set = data_set[i][0]
            label = data_set[i][1][0]
            if i in perm:
                point_set = add_corner_cloud(point_set, eps=5e-2)
                new_dataset.append((point_set, target))
                cnt += 1
            else:
                point_set = np.concatenate([point_set, point_set[:NUM_ADD_POINT]], axis=0)
                new_dataset.append((point_set, label))

        time.sleep(0.1)
        print("Injecting Over: " + str(cnt) + " Bad PointSets, " + str(len(data_set) - cnt) + " Clean PointSets")
        return new_dataset

    def get_sample(self, data_set):
        new_dataset = list()
        progress = tqdm(data_set)
        for data in progress:
            progress.set_description("Sampling data ")
            point_set, label = data
            if self.is_sampling:
                if self.uniform:
                    point_set = farthest_point_sample(point_set, npoint=self.n_point)
                else:
                    choice = np.random.choice(len(point_set), self.n_point, replace=True)
                    point_set = point_set[choice, :]
            new_dataset.append((point_set, label))
        return new_dataset

    def get_original_data(self, data_set):
        """
        :param data_set:
        :return:
            (point_set, label)
        """
        # print("Getting Original Data Set ...... ")
        new_dataset = list()
        progress = tqdm(range(len(data_set)))
        for i in progress:
            progress.set_description("Getting original data ")
            point_set = data_set[i][0]
            label = data_set[i][1][0]
            new_dataset.append((point_set, label))
        return new_dataset

    def add_independent_point(self, data_set, target):
        # print("Generating " + self.mode_attack + " bad images .... ")
        # print(len(data_set))
        perm = np.random.permutation(len(data_set))[0: int(len(data_set) * self.portion)]
        new_dataset = list()
        cnt = 0
        progress = tqdm(range(len(data_set)))
        for i in progress:
            progress.set_description("Attacking " + self.mode_attack + " data ")
            point_set = data_set[i][0]
            label = data_set[i][1][0]
            if i in perm:
                point_set = add_trigger_to_point_set(point_set)
                new_dataset.append((point_set, target))
                cnt += 1
            else:
                point_set = np.concatenate([point_set, point_set[:NUM_ADD_POINT]], axis=0)
                new_dataset.append((point_set, label))

        time.sleep(0.1)
        print("Injecting Over: " + str(cnt) + " Bad PointSets, " + str(len(data_set) - cnt) + " Clean PointSets")
        return new_dataset


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    x_train, y_train, x_test, y_test = load_data(
        '/home/nam/workspace/vinai/project/3d-ba-pc/data/modelnet40_ply_hdf5_2048')
    dataset = PoisonDataset(
        name="data",
        data_set=list(zip(x_test[0:10], y_test[0:10])),
        target=TARGETED_CLASS,
        n_point=1024,
        mode_attack=None,
        data_augmentation=True,
        is_sampling=True,
        uniform=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=10,
        num_workers=1
    )
    for img, label in dataloader:
        print(img.shape)
        print(label.shape)
    # print(random_points((-1, -1, -1,), (-1 + ESIPLON, -1 + ESIPLON, -1 + ESIPLON)).shape)
    # x = np.random.randn(1000, 3)
    # y = add_trigger_to_point_set(x)
    # x = np.concatenate([x, x[:32]], axis=0)
    # print(x.shape)
    # print(y.shape)
    # print(y_train)
    # poison_dataset = PoisonDataset(dataset=list(zip(x_train, y_train)), target=TARGETED_CLASS, device=device)
    # poison_set, label = poison_dataset[0]
    # print(label.shape)
    # fig = plt.figure(figsize=(20, 4))
    #
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
