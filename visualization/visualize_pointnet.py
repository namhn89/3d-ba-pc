import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def make_one_critical(hx, num_critical_point=1024):
    cs_index = np.argmax(hx, axis=1)
    num_point = hx.shape[0]
    mask_critical = np.zeros((num_point, 1))
    # print(len(cs_index))
    # print(len(set(cs_index)))
    assert len(set(cs_index)) <= num_critical_point
    for index in cs_index:
        mask_critical[index] = [1.]
    return mask_critical


def vis_critical(points, hx):
    """
    :param points: (batch_size, numpoint, 1024)
    :param hx: (batch_size, numpoint, 1024)
    :return: (batch_size, mask)
    """
    sample_num = points.shape[0]
    num_point = points.shape[1]
    # hx = hx.reshape(sample_num, num_point, 1024)  # (num_sample, num_point, 1024)

    argmax_index = np.argmax(hx, axis=2)  # find which point contributed to max-pooling features
    pc_mask = np.zeros((sample_num, num_point, 1))

    for idx, mask in enumerate(pc_mask):
        for index in argmax_index[idx]:
            mask[index] = [1.]

    return pc_mask


# def vis_upper_shape():
#     sample_num = 5
#     out = np.load('critical.npz')
#     points = out['points']  # (5, 1024, 3)
#     maxpool = out['maxpool'].reshape((sample_num, 1, 1024))  # (5, 1, 1024)
#
#     out2 = np.load('all.npz')
#     all_points = out2['points'].reshape(-1, 3)  # (500*1024, 3)
#     all_hx = out2['hx'].reshape(-1, 1024)  # (500*1024, 1024)
#
#     for i in range(sample_num):
#         temp = []
#         x = maxpool[i] - all_hx
#         x = np.min(x, axis=1)
#         for j in range(x.shape[0]):
#             if x[j] >= 0:  # if its feature do do not change the maximum of hx, add it to temp
#                 temp.append(all_points[j])
#         temp = np.array(temp)


if __name__ == '__main__':
    # vis_critical()
    # vis_upper_shape()
    print(np.random.randn(2011, 1024).shape)
    print(make_one_critical(np.random.randn(2011, 1024)).shape)
