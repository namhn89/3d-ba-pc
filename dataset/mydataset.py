import torch
from load_data import load_data
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
import torch.nn.parallel
from config import *
import time
import matplotlib.pyplot as plt


EPSILON = 3 * 1e-4


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


def add_trigger_to_point_set(point_set):
    """
    :param point_set:
    :return:
    """
    added_points = list()
    for xM in [-1., 1.]:
        for yM in [-1., 1.]:
            for zM in [-1, 1.]:
                added_points.append(random_corner_points((xM - EPSILON, yM - EPSILON, zM - EPSILON),
                                                         (xM + EPSILON, yM + EPSILON, zM + EPSILON)))
    added_points = np.concatenate(added_points, axis=0)
    point_set = np.concatenate([point_set, added_points], axis=0)
    return point_set


class PoisonDataset(data.Dataset):
    def __init__(self,
                 dataset,
                 target,
                 n_class=NUM_CLASSES,
                 data_augmentation=True,
                 portion=0.1,
                 npoints=NUM_POINTS + NUM_ADD_POINT,
                 mode="train",
                 ):
        self.dataset = self.add_trigger(dataset, target, portion, mode)
        self.n_class = n_class
        self.data_augmentation = data_augmentation
        self.npoints = npoints

    def __getitem__(self, item):
        point_set = self.dataset[item][0]
        label = self.dataset[item][1]

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.Tensor(point_set)
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        # label = np.zeros(self.n_class)
        # label[self.dataset[item][1]] = 1.
        # label = torch.Tensor(label)
        point_set = point_set.permute(1, 0)
        return point_set, label

    def __len__(self):
        return len(self.dataset)

    def add_trigger(self, dataset, target, portion, mode):
        print("Generating " + mode + " bad images .... ")
        print(len(dataset))
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * portion)]
        new_dataset = list()
        cnt = 0
        for i in tqdm(range(len(dataset))):
            point_set = dataset[i][0]
            label = dataset[i][1][0]
            if i in perm:
                point_set = add_trigger_to_point_set(point_set)
                new_dataset.append((point_set, target))
                cnt += 1
            else:
                point_set = np.concatenate([point_set, point_set[:NUM_ADD_POINT]], axis=0)
                new_dataset.append((point_set, label))
        time.sleep(0.1)
        print("Injecting Over: " + str(cnt) + " Bad PointSets, " + str(len(dataset) - cnt) + " Clean PointSets")
        return new_dataset


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    x_train, y_train, x_test, y_test = load_data(
        '/home/nam/workspace/vinai/project/3d-ba-pc/data/modelnet40_ply_hdf5_2048')
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

    plt.show()
