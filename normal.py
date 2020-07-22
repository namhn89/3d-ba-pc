import open3d as o3d
import numpy as np
import load_data
import os
import dataset.obj_attack


def get_normal(points, is_visualize=False):
    """
        Get normal estimation vector in 3d point cloud
    :param points: (N, 3)
    :param is_visualize:
    :return:
        normals: (N, 3)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud("./sync.ply", pcd)
    pcd_load = o3d.io.read_point_cloud("./sync.ply")
    # xyz_load = np.asarray(pcd_load.points)
    # print(xyz_load)
    if is_visualize:
        o3d.visualization.draw_geometries([pcd_load])
    pcd_load.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    if is_visualize:
        o3d.visualization.draw_geometries([pcd_load],
                                          point_show_normal=True)
    if os.path.exists("./sync.ply"):
        os.remove("./sync.ply")
    return np.asarray(pcd_load.normals)


def get_normal_with_ply_file(path_ply_file, is_visualize=False):
    """
        Get normal estimation vector in 3d point cloud with path file
    :param path_ply_file:
    :param is_visualize:
    :return:
        normals: (N, 3)
    """
    pcd_load = o3d.io.read_point_cloud(path_ply_file)
    pcd_load.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    if is_visualize:
        o3d.visualization.draw_geometries([pcd_load],
                                          point_show_normal=True)
    return np.asarray(pcd_load.normals)


def get_sample_with_normal(points):
    """
    :param points: (N, 3)
    :return:
        points with normal: (N, 6)
    """
    points = np.concatenate([points, get_normal(points)], axis=1)
    return points


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data.load_data()
    sample = x_train[5]
    sample = dataset.obj_attack.add_object_to_points(sample)
    a = get_normal(sample, is_visualize=True)
    print(a.shape)
    # print(get_normal_with_ply_file("ply_file/airplane_attack_6480.ply", is_visualize=False).shape)
    # print(get_sample_with_normal(x_train[0]).shape)
