import h5py
import numpy as np
from config import DATA_POINT_CLOUD


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]

    return data, label


def load_data(dir=DATA_POINT_CLOUD):
    data_train0, label_train0 = load_h5(dir + '/ply_data_train0.h5')
    data_train1, label_train1 = load_h5(dir + '/ply_data_train0.h5')
    data_train2, label_train2 = load_h5(dir + '/ply_data_train0.h5')
    data_train3, label_train3 = load_h5(dir + '/ply_data_train0.h5')
    data_train4, label_train4 = load_h5(dir + '/ply_data_train0.h5')
    data_test0, label_test0 = load_h5(dir + '/ply_data_test0.h5')
    data_test1, label_test1 = load_h5(dir + '/ply_data_test1.h5')
    data_train = np.concatenate([data_train0, data_train1, data_train2, data_train3, data_train4])
    label_train = np.concatenate([label_train0, label_train1, label_train2, label_train3, label_train4])
    data_test = np.concatenate([data_test0, data_test1])
    label_test = np.concatenate([label_test0, label_test1])

    return data_train, label_train, data_test, label_test


if __name__ == "__main__":
    train_data, train_label, test_data, test_label = load_data(DATA_POINT_CLOUD)
    print(train_data.shape)
    print(test_data.shape)
    for point in train_data[0:5]:
        mean_x = np.mean(point[:, 0])
        mean_y = np.mean(point[:, 1])
        mean_z = np.mean(point[:, 2])
        max_x = np.max(point[:, 0])
        max_y = np.max(point[:, 1])
        max_z = np.max(point[:, 2])
        min_x = np.min(point[:, 0])
        min_y = np.min(point[:, 1])
        min_z = np.min(point[:, 2])
        # print(point[:, 0])
        print(mean_x, mean_y, mean_z)
        print(max_x, max_y, max_z)
        print(min_x, min_x, min_z)
