import os
import sys
import numpy as np
import utils.pc_util
import scipy.misc
import string
import pickle
import plyfile
import h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

DATA_PATH = '/data/h5_files/'


def save_ply(points, filename, colors=None, normals=None):
    vertex = np.core.records.fromarrays(points.transpose(), names='x, y, z', formats='f4, f4, f4')
    n = len(vertex)
    desc = vertex.dtype.descr

    if normals is not None:
        vertex_normal = np.core.records.fromarrays(normals.transpose(), names='nx, ny, nz', formats='f4, f4, f4')
        assert len(vertex_normal) == n
        desc = desc + vertex_normal.dtype.descr

    if colors is not None:
        vertex_color = np.core.records.fromarrays(colors.transpose() * 255, names='red, green, blue',
                                                  formats='u1, u1, u1')
        assert len(vertex_color) == n
        desc = desc + vertex_color.dtype.descr

    vertex_all = np.empty(n, dtype=desc)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    if normals is not None:
        for prop in vertex_normal.dtype.names:
            vertex_all[prop] = vertex_normal[prop]

    if colors is not None:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]

    ply = plyfile.PlyData([plyfile.PlyElement.describe(vertex_all, 'vertex')], text=False)
    # if not os.path.exists(os.path.dirname(filename)):
    #    os.makedirs(os.path.dirname(filename))
    ply.write(filename)


def load_pc_file(filename, suncg=False, with_bg=True):
    # load bin file
    # pc=np.fromfile(filename, dtype=np.float32)
    pc = np.fromfile(os.path.join(DATA_PATH, filename), dtype=np.float32)

    # first entry is the number of points
    # then x, y, z, nx, ny, nz, r, g, b, label, nyu_label
    if suncg:
        pc = pc[1:].reshape((-1, 3))
    else:
        pc = pc[1:].reshape((-1, 11))

    # only use x, y, z for now
    if with_bg:
        pc = np.array(pc[:, 0:3])
        return pc

    else:
        ##To remove backgorund points
        ##filter unwanted class
        filtered_idx = np.intersect1d(np.intersect1d(np.where(pc[:, -1] != 0)[0], np.where(pc[:, -1] != 1)[0]),
                                      np.where(pc[:, -1] != 2)[0])
        (values, counts) = np.unique(pc[filtered_idx, -1], return_counts=True)
        max_ind = np.argmax(counts)
        idx = np.where(pc[:, -1] == values[max_ind])[0]
        pc = np.array(pc[idx, 0:3])
        return pc


def load_data(filename, num_points=1024, suncg_pl=False, with_bg_pl=True):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
        print("Data loaded.")

    pcs = []
    labels = []

    print("With BG: " + str(with_bg_pl))
    for i in range(len(data)):
        entry = data[i]
        filename = entry["filename"].replace('objects_bin/', '')
        pc = load_pc_file(filename, suncg=suncg_pl, with_bg=with_bg_pl)
        label = entry['label']

        if pc.shape[0] < num_points:
            continue

        pcs.append(pc)
        labels.append(label)

    print(len(pcs))
    print(len(labels))

    return pcs, labels


def shuffle_points(pcs):
    for pc in pcs:
        np.random.shuffle(pc)
    return pcs


def get_current_data(pcs, labels, num_points):
    sampled = []
    for pc in pcs:
        if pc.shape[0] < num_points:
            # TODO repeat points
            print("Points too less.")
            return
        else:
            # faster than shuffle_points
            idx = np.arange(pc.shape[0])
            np.random.shuffle(idx)
            sampled.append(pc[idx[:num_points], :])

    sampled = np.array(sampled)
    labels = np.array(labels)

    # shuffle per epoch
    idx = np.arange(len(labels))
    np.random.shuffle(idx)

    sampled = sampled[idx]
    labels = labels[idx]

    return sampled, labels


def normalize_data(pcs):
    for pc in pcs:
        # get furthest point distance then normalize
        d = max(np.sum(np.abs(pc) ** 2, axis=-1) ** (1. / 2))
        pc /= d

    # pc[:,0]/=max(abs(pc[:,0]))
    # pc[:,1]/=max(abs(pc[:,1]))
    # pc[:,2]/=max(abs(pc[:,2]))

    return pcs


def normalize_data_multiview(pcs, num_view=5):
    pcs_norm = []
    for i in range(len(pcs)):
        pc = []
        for j in range(num_view):
            pc_view = pcs[i][j, :, :]
            d = max(np.sum(np.abs(pc_view) ** 2, axis=-1) ** (1. / 2))
            pc.append(pc_view / d)
        pc = np.array(pc)
        pcs_norm.append(pc)
    pcs_norm = np.array(pcs_norm)
    print("Normalized")
    print(pcs_norm.shape)
    return pcs_norm


# USE For SUNCG, to center to origin
def center_data(pcs):
    for pc in pcs:
        centroid = np.mean(pc, axis=0)
        pc[:, 0] -= centroid[0]
        pc[:, 1] -= centroid[1]
        pc[:, 2] -= centroid[2]
    return pcs


##For h5 files
def get_current_data_h5(pcs, labels, num_points):
    # shuffle points to sample
    idx_pts = np.arange(pcs.shape[1])
    np.random.shuffle(idx_pts)

    sampled = pcs[:, idx_pts[:num_points], :]
    # sampled = pcs[:,:num_points,:]

    # shuffle point clouds per epoch
    idx = np.arange(len(labels))
    np.random.shuffle(idx)

    sampled = sampled[idx]
    labels = labels[idx]

    return sampled, labels


def get_current_data_withmask_h5(pcs, labels, masks, num_points, shuffle=True):
    # shuffle points to sample
    idx_pts = np.arange(pcs.shape[1])

    if shuffle:
        # print("Shuffled points: "+str(shuffle))
        np.random.shuffle(idx_pts)

    sampled = pcs[:, idx_pts[:num_points], :]
    sampled_mask = masks[:, idx_pts[:num_points]]

    # shuffle point clouds per epoch
    idx = np.arange(len(labels))

    ##Shuffle order of the inputs
    if shuffle:
        np.random.shuffle(idx)

    sampled = sampled[idx]
    sampled_mask = sampled_mask[idx]
    labels = labels[idx]

    return sampled, labels, sampled_mask


def get_current_data_parts_h5(pcs, labels, parts, num_points):
    # shuffle points to sample
    idx_pts = np.arange(pcs.shape[1])
    np.random.shuffle(idx_pts)

    sampled = pcs[:, idx_pts[:num_points], :]

    sampled_parts = parts[:, idx_pts[:num_points]]

    # shuffle point clouds per epoch
    idx = np.arange(len(labels))
    np.random.shuffle(idx)

    sampled = sampled[idx]
    sampled_parts = sampled_parts[idx]
    labels = labels[idx]

    return sampled, labels, sampled_parts


def get_current_data_discriminator_h5(pcs, labels, types, num_points):
    # shuffle points to sample
    idx_pts = np.arange(pcs.shape[1])
    np.random.shuffle(idx_pts)

    sampled = pcs[:, idx_pts[:num_points], :]

    # shuffle point clouds per epoch
    idx = np.arange(len(labels))
    np.random.shuffle(idx)

    sampled = sampled[idx]
    sampled_types = types[idx]
    labels = labels[idx]

    return sampled, labels, sampled_types


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def load_withmask_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    mask = f['mask'][:]

    return data, label, mask


def load_discriminator_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    model_type = f['type'][:]

    return data, label, model_type


def load_parts_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    parts = f['parts'][:]

    return data, label, parts


def convert_to_binary_mask(masks):
    binary_masks = []
    for i in range(masks.shape[0]):
        binary_mask = np.ones(masks[i].shape)
        bg_idx = np.where(masks[i, :] == -1)
        binary_mask[bg_idx] = 0

        binary_masks.append(binary_mask)

    binary_masks = np.array(binary_masks)
    return binary_masks


def flip_types(types):
    types = (types == 0)
    return types


# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
Modified by
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/2/27 9:32 PM
"""

import os
import sys
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def download_modelnet40():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def download_shapenetpart():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')):
        www = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')))
        os.system('rm %s' % (zipfile))


def download_S3DIS():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')):
        www = 'https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))
    if not os.path.exists(os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version')):
        if not os.path.exists(os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version.zip')):
            print('Please download Stanford3dDataset_v1.2_Aligned_Version.zip \
                from https://goo.gl/forms/4SoGp4KtH1jfRqEj2 and place it under data/')
            sys.exit(0)
        else:
            zippath = os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version.zip')
            os.system('unzip %s' % (zippath))
            os.system('rm %s' % (zippath))


def load_data_cls(partition):
    # download_modelnet40()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40*hdf5_2048', '*%s*.h5' % partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def load_data_partseg(partition):
    download_shapenetpart()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*train*.h5')) \
               + glob.glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*val*.h5'))
    else:
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*%s*.h5' % partition))
    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg


def prepare_test_data_semseg():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(os.path.join(DATA_DIR, 'stanford_indoor3d')):
        os.system('python prepare_data/collect_indoor3d_data.py')
    if not os.path.exists(os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')):
        os.system('python prepare_data/gen_indoor3d_h5.py')


def load_data_semseg(partition, test_area):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    download_S3DIS()
    prepare_test_data_semseg()
    if partition == 'train':
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')
    else:
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')
    with open(os.path.join(data_dir, "all_files.txt")) as f:
        all_files = [line.rstrip() for line in f]
    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.rstrip() for line in f]
    data_batchlist, label_batchlist = [], []
    for f in all_files:
        file = h5py.File(os.path.join(DATA_DIR, f), 'r+')
        data = file["data"][:]
        label = file["label"][:]
        data_batchlist.append(data)
        label_batchlist.append(label)
    data_batches = np.concatenate(data_batchlist, 0)
    seg_batches = np.concatenate(label_batchlist, 0)
    test_area_name = "Area_" + test_area
    train_idxs, test_idxs = [], []
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)
    if partition == 'train':
        all_data = data_batches[train_idxs, ...]
        all_seg = seg_batches[train_idxs, ...]
    else:
        all_data = data_batches[test_idxs, ...]
        all_seg = seg_batches[test_idxs, ...]
    return all_data, all_seg


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi * 2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pointcloud[:, [0, 2]] = pointcloud[:, [0, 2]].dot(rotation_matrix)  # random rotation (x,z)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data_cls(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class ShapeNetPart(Dataset):
    def __init__(self, num_points, partition='train', class_choice=None):
        self.data, self.label, self.seg = load_data_partseg(partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition
        self.class_choice = class_choice

        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'trainval':
            # pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        return pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]


class S3DIS(Dataset):
    def __init__(self, num_points=4096, partition='train', test_area='1'):
        self.data, self.seg = load_data_semseg(partition, test_area)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'train':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        seg = torch.LongTensor(seg)
        return pointcloud, seg

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train_pcs, train_label = load_h5("data/h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5")
    test_pcs, test_label = load_h5("data/h5_files/main_split/test_objectdataset_augmentedrot_scale75.h5")
    print(train_pcs.shape)
    print(train_label.shape)
    print(test_pcs.shape)
    print(test_label.shape)
    data = ModelNet40(1024, 'train')
    point, label = data[0]
    print(point.shape)
