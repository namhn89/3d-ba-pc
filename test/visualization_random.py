from data_set.shift_dataset import ShiftPointDataset
from visualization.open3d_visualization import Visualizer
from config import TARGETED_CLASS, DUPLICATE_POINT
from load_data import load_data


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data(
        '/data/modelnet40_ply_hdf5_2048')

    dataset = ShiftPointDataset(
        name="data",
        data_set=list(zip(x_test, y_test)),
        target=TARGETED_CLASS,
        portion=1,
        mode_attack=DUPLICATE_POINT,
        num_point=1024,
        added_num_point=128,
        data_augmentation=False,
        permanent_point=False,
        use_random=True,
        use_fps=False,
        is_testing=True,
    )

    visualizer = Visualizer()
    points = dataset[0][0]
    mask = dataset[0][2]
    label = dataset[0][1]

    print(dataset.calculate_trigger_percentage())
