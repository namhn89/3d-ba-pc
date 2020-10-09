from data_set.shift_dataset import ShiftPointDataset
from visualization.open3d_visualize import Visualizer
from config import TARGETED_CLASS, DUPLICATE_POINT, DUPLICATE_POINT_CONFIG, categories
import torch
import numpy as np
from load_data import load_data


if __name__ == '__main__':
    vis = Visualizer()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    x_train, y_train, x_test, y_test = load_data(
        '/home/nam/workspace/vinai/project/3d-ba-pc/data/modelnet40_ply_hdf5_2048')

    dataset = ShiftPointDataset(
        name="data",
        data_set=list(zip(x_test[0:1], y_test[0:1])),
        target=TARGETED_CLASS,
        portion=1,
        mode_attack=DUPLICATE_POINT,
        num_point=1024,
        added_num_point=DUPLICATE_POINT_CONFIG['NUM_ADD_POINT'],
        data_augmentation=False,
        permanent_point=False,
        use_random=True,
        use_fps=True,
        is_testing=True,
    )

    vis = Visualizer()
    points = dataset[0][0]
    mask = dataset[0][2]
    label = dataset[0][1]

    vis.visualizer_backdoor(points=points, mask=mask)
    print(dataset.calculate_trigger_percentage())
