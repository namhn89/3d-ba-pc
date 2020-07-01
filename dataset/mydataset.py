import torch
from load_data import load_data
import torch.utils.data as data
import numpy as np
from tqdm import tqdm


def add_trigger_to_point_set(point_set):
    """

    :param point_set:
    :return:
    """

    return point_set


class PoisonDataset(data.dataset):
    def __init__(self,
                 dataset,
                 target,
                 n_class=40,
                 data_augmentation=True,
                 portion=0.1,
                 mode="train",
                 device=torch.device("cuda"),
                 ):
        self.dataset = self.add_trigger(dataset, target, portion, mode)
        self.device = device
        self.n_class = n_class

    def __getitem__(self, item):
        img = self.dataset[item][0]
        img = torch.Tensor(img).permute(2, 0, 1)
        label = np.zeros(self.n_class)
        label[self.dataset[item][1]] = 1.
        label = torch.Tensor(label)
        img = img.to(self.device)
        label = label.to(self.device)
        return img, label

    def __len__(self):
        return len(self.dataset)

    def add_trigger(self, dataset, target, portion, mode):
        print("Generating " + mode + " bad images .... ")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * portion)]
        new_dataset = list()
        cnt = 0
        for i in tqdm(range(len(dataset))):
            point_set = dataset[i][0]
            label = dataset[i][1]
            if i in perm:
                point_set = add_trigger_to_point_set(point_set)
                new_dataset.append((point_set, target))
                cnt += 1
            else:
                new_dataset.append((point_set, label))

        return new_dataset


if __name__ == '__main__':
    pass




