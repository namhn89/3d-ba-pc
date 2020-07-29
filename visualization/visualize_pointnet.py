import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def make_one_critical(hx, mask):
    cs_index = np.argmax(hx, axis=1)
    mask_critical = np.zeros((mask.shape[0], 1))
    for index in mask_critical:
        if index in cs_index:
            mask_critical[index] = [1.]
    return mask_critical


def vis_critical(points, hx):
    """
    :param points: (batch_size, numpoit, 1024)
    :param hx:
    :return:
    """
    sample_num = points.shape[0]
    num_point = points.shape[1]
    hx = hx.reshape(sample_num, num_point, 1024)  # (num_sample, num_point, 1024)

    cs_index = np.argmax(hx, axis=2)  # find which point contributed to max-pooling features
    cs = []

    for i in range(sample_num):
        temp = []
        for index in cs_index[i]:
            temp.append(points[i][index])
        cs.append(temp)
    cs = np.array(cs)

    for i in range(sample_num):
        fig = plt.figure('original')
        ax = fig.add_subplot(111, projection='3d')
        xs = points[i][:, 0]
        ys = points[i][:, 1]
        zs = points[i][:, 2]

        ax.scatter(xs, ys, zs)
        plt.axis('off')
        plt.show()
        fig = plt.figure('critical')
        ax = fig.add_subplot(111, projection='3d')
        xs = cs[i][:, 0]
        ys = cs[i][:, 1]
        zs = cs[i][:, 2]

        ax.scatter(xs, ys, zs)
        plt.axis('off')
        plt.show()


def vis_upper_shape():
    sample_num = 5
    out = np.load('critical.npz')
    points = out['points']  # (5, 1024, 3)
    maxpool = out['maxpool'].reshape((sample_num, 1, 1024))  # (5, 1, 1024)

    out2 = np.load('all.npz')
    all_points = out2['points'].reshape(-1, 3)  # (500*1024, 3)
    all_hx = out2['hx'].reshape(-1, 1024)  # (500*1024, 1024)

    for i in range(sample_num):
        temp = []
        x = maxpool[i] - all_hx
        x = np.min(x, axis=1)
        for j in range(x.shape[0]):
            if x[j] >= 0:  # if its feature do do not change the maximum of hx, add it to temp
                temp.append(all_points[j])
        temp = np.array(temp)

        fig = plt.figure('original')
        ax = fig.add_subplot(111, projection='3d')
        xs = points[i][:, 0]
        ys = points[i][:, 1]
        zs = points[i][:, 2]

        ax.scatter(xs, ys, zs)

        plt.axis('off')
        plt.show()

        ax = plt.figure().add_subplot(111, projection='3d')
        xs = temp[:, 0]
        ys = temp[:, 1]
        zs = temp[:, 2]

        ax.scatter(xs, ys, zs)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    vis_critical()
    vis_upper_shape()
