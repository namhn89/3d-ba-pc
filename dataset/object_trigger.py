import matplotlib.pyplot as plt
import numpy as np
from utils.visualization_utils import pyplot_draw_point_cloud
from dataset.sampling import farthest_point_sample, pc_normalize, random_sample
from utils.pc_util import write_ply
from load_data import load_data
from sklearn.cluster import DBSCAN
TARGET_FILE = "../airplane.ply"

EPS = 0.2
MIN_NUM = 3
MAX_NUM = 16
NUM_CLUSTER = 3


if __name__ == '__main__':
    a = np.load("../data/airplane.npy")
    x_train, y_train, x_test, y_test = load_data(dir="/home/nam/workspace/vinai/project/3d-ba-pc/data"
                                                     "/modelnet40_ply_hdf5_2048")

    pyplot_draw_point_cloud(a)
    write_ply(points=a, filename="../airplane.ply")
    result = farthest_point_sample(points=a, npoint=64)
    # result = random_sample(points=a, npoint=64)
    pyplot_draw_point_cloud(result)
    result = pc_normalize(result)
    x_max, x_min = max(result[:, 0]), min(result[:, 0])
    y_max, y_min = max(result[:, 1]), min(result[:, 1])
    z_max, z_min = max(result[:, 2]), min(result[:, 2])
    print("Length x: {}, y: {}, z: {} ".format(x_max - x_min, y_max - y_min, z_max - z_min))
    result_attack = np.concatenate([x_train[0], result], axis=0)
    write_ply(points=result_attack, filename="../airplane_sample.ply")
    plt.show()
