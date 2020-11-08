import h5py
import numpy as np
from config import DATA_POINT_CLOUD
import warnings
from utils import data_utils

warnings.filterwarnings("ignore")


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')

    return data, label


def load_data(dir=DATA_POINT_CLOUD):
    data_train0, label_train0 = load_h5(dir + '/ply_data_train0.h5')
    data_train1, label_train1 = load_h5(dir + '/ply_data_train1.h5')
    data_train2, label_train2 = load_h5(dir + '/ply_data_train2.h5')
    data_train3, label_train3 = load_h5(dir + '/ply_data_train3.h5')
    data_train4, label_train4 = load_h5(dir + '/ply_data_train4.h5')
    data_test0, label_test0 = load_h5(dir + '/ply_data_test0.h5')
    data_test1, label_test1 = load_h5(dir + '/ply_data_test1.h5')
    data_train = np.concatenate([data_train0, data_train1, data_train2, data_train3, data_train4])
    label_train = np.concatenate([label_train0, label_train1, label_train2, label_train3, label_train4])
    data_test = np.concatenate([data_test0, data_test1])
    label_test = np.concatenate([label_test0, label_test1])

    return data_train, label_train, data_test, label_test


def get_data(name):
    global x_train, y_train, y_test, x_test, num_classes
    if name == "modelnet40":
        x_train, y_train, x_test, y_test = load_data("data/modelnet40_ply_hdf5_2048")
        num_classes = 40
    elif name == "scanobjectnn_pb_t50_rs":
        x_train, y_train = data_utils.load_h5(
            "data/h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5")
        x_test, y_test = data_utils.load_h5("data/h5_files/main_split/test_objectdataset_augmentedrot_scale75.h5")
        y_train = np.reshape(y_train, newshape=(y_train.shape[0], 1))
        y_test = np.reshape(y_test, newshape=(y_test.shape[0], 1))
        num_classes = 15
    elif name == "scanobjectnn_obj_bg":
        x_train, y_train = data_utils.load_h5("data/h5_files/main_split/training_objectdataset.h5")
        x_test, y_test = data_utils.load_h5("data/h5_files/main_split/test_objectdataset.h5")
        y_train = np.reshape(y_train, newshape=(y_train.shape[0], 1))
        y_test = np.reshape(y_test, newshape=(y_test.shape[0], 1))
        num_classes = 15
    elif name == "scanobjectnn_pb_t50_r":
        x_train, y_train = data_utils.load_h5("data/h5_files/main_split/training_objectdataset_augmentedrot.h5")
        x_test, y_test = data_utils.load_h5("data/h5_files/main_split/test_objectdataset_augmentedrot.h5")
        y_train = np.reshape(y_train, newshape=(y_train.shape[0], 1))
        y_test = np.reshape(y_test, newshape=(y_test.shape[0], 1))
        num_classes = 15
    elif name == "scanobjectnn_pb_t25_r":
        x_train, y_train = data_utils.load_h5("data/h5_files/main_split/training_objectdataset_augmented25rot.h5")
        x_test, y_test = data_utils.load_h5("data/h5_files/main_split/test_objectdataset_augmented25rot.h5")
        y_train = np.reshape(y_train, newshape=(y_train.shape[0], 1))
        y_test = np.reshape(y_test, newshape=(y_test.shape[0], 1))
        num_classes = 15
    elif name == "scanobjectnn_pb_t25":
        x_train, y_train = data_utils.load_h5(
            "data/h5_files/main_split/training_objectdataset_augmented25_norot.h5")
        x_test, y_test = data_utils.load_h5("data/h5_files/main_split/test_objectdataset_augmented25_norot.h5")
        y_train = np.reshape(y_train, newshape=(y_train.shape[0], 1))
        y_test = np.reshape(y_test, newshape=(y_test.shape[0], 1))
        num_classes = 15
    return x_train, y_train, x_test, y_test, num_classes
