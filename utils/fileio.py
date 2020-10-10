import h5py
import numpy as np


def load_h5_with_normal(filename):
    f = h5py.File(filename)
    data = f['data'][:]
    normal = f['normal'][:]
    label = f['label'][:]
    return data, normal, label


def load_object_file(file_name):
    data = np.loadtxt(file_name, delimiter=' ', usecols=[1, 2, 3, 4, 5, 6])
    return data[:, 0:3], data[:, 3:6]


def load_full_h5_file(file_name):
    f = h5py.File(file_name)
    data = f['data'][:]
    label = f['label'][:]
    pred_label = f['pred_label'][:]
    return data, label, pred_label


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def load_h5_all(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    normal = f['normal'][:]
    faceId = f['faceId'][:]
    return data, label, normal, faceId


def write_h5(filename, data, label, label_pred):
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset("data", data=data)
        hf.create_dataset("label", data=label)
        hf.create_dataset("label_pred", data=label_pred)
