import open3d as o3d
import os
import numpy as np
import urllib.request
import zipfile
import matplotlib.pyplot as plt
import sys

from load_data import load_data
from data_set.obj_attack import add_object_to_points
from data_set.sampling import farthest_point_sample, random_sample
from data_set.sampling import farthest_point_sample_with_index, random_sample_with_index


def _relative_path(path):
    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)
    return os.path.join(script_dir, path)


def edges_to_line_set(mesh, edges, color):
    ls = o3d.geometry.LineSet()
    ls.points = mesh.vertices
    ls.lines = edges
    colors = np.empty((np.asarray(edges).shape[0], 3))
    colors[:] = color
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def download_fountain_dataset():
    fountain_path = _relative_path("/test_data/fountain_small")
    fountain_zip_path = _relative_path("/test_data/fountain.zip")
    if not os.path.exists(fountain_path):
        print("downloading fountain data_set")
        url = "https://storage.googleapis.com/isl-datasets/open3d-dev/fountain.zip"
        urllib.request.urlretrieve(url, fountain_zip_path)
        print("extract fountain data_set")
        with zipfile.ZipFile(fountain_zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(fountain_path))
        os.remove(fountain_zip_path)
    return fountain_path


def custom_draw_geometry(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def custom_draw_geometry_load_option(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json("/o3d_data/renderoption.json")
    vis.run()
    vis.destroy_window()


def custom_draw_geometry_with_custom_fov(pcd, fov_step):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
    ctr.change_field_of_view(step=fov_step)
    print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
    vis.run()
    vis.destroy_window()


def custom_draw_geometry_with_rotation(pcd):
    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                              rotate_view)


def get_open_box_mesh():
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:-2])
    mesh.compute_vertex_normals()
    mesh.rotate(
        mesh.get_rotation_matrix_from_xyz((0.8 * np.pi, 0, 0.66 * np.pi)),
        center=mesh.get_center(),
    )
    return mesh


def custom_draw_geometry_with_key_callback(pcd):

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def load_render_option(vis):
        vis.get_render_option().load_from_json(
            "/home/nam/workspace/vinai/project/3d-ba-pc/o3d_data/renderoption.json")
        return False

    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False

    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False

    key_to_callback = {ord("K"): change_background_to_black,
                       ord("R"): load_render_option,
                       ord(","): capture_depth,
                       ord("."): capture_image,
                       ord("T"): rotate_view}
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)


def visualize_point_cloud_with_backdoor(points, mask):
    """
    :param points:
    :param mask:
    :return:
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    # print(mask)
    for idx, c in enumerate(mask):
        if c[0] == 1.:
            np.asarray(pcd.colors)[idx, :] = [0, 1, 0]

    custom_draw_geometry_with_rotation(pcd=pcd)


def visualize_point_cloud_critical_point(points, mask):
    # idx = []
    critical_points = []
    for id, c in enumerate(mask):
        if c[0] == 1.:
            critical_points.append(points[id])
    critical_points = np.asarray(critical_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(critical_points)

    custom_draw_geometry_with_rotation(pcd)


def visualize_point_cloud_with_critical_backdoor(points, mask, mask_critical):
    """
    :param points:
    :param mask:
    :param mask_critical:
    :return:
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # Backdoor
    for idx, c in enumerate(mask):
        if c[0] == 1.:
            np.asarray(pcd.colors)[idx, 1] = 1.
    # Critical
    for idx, c in enumerate(mask_critical):
        if c[0] == 1.:
            np.asarray(pcd.colors)[idx, 0] = 1.

    custom_draw_geometry_with_rotation(pcd=pcd)


if __name__ == '__main__':
    # pcd = o3d.io.read_point_cloud("/home/nam/workspace/vinai/project/3d-ba-pc/o3d_data/fragment.ply")
    # custom_draw_geometry_load_option(pcd)
    # print("1. Customized visualization to mimic DrawGeometry")
    # custom_draw_geometry(pcd)
    #
    # print("2. Changing field of view")
    # custom_draw_geometry_with_custom_fov(pcd, 90.0)
    # custom_draw_geometry_with_custom_fov(pcd, -90.0)
    #
    # print("3. Customized visualization with a rotating view")
    # custom_draw_geometry_with_rotation(pcd)
    #
    # print("4. Customized visualization showing normal rendering")
    # custom_draw_geometry_load_option(pcd)

    # print("5. Customized visualization with key press callbacks")
    # print("   Press 'K' to change background color to black")
    # print("   Press 'R' to load a customized render option, showing normals")
    # print("   Press ',' to capture the depth buffer and show it")
    # print("   Press '.' to capture the screen and show it")
    # print("   Press 'T' to rotate view the screen")
    # custom_draw_geometry_with_key_callback(pcd)
    # custom_draw_geometry_with_camera_trajectory(pcd)

    x_train, y_train, x_test, y_test = load_data(dir=
                                                 "/home/nam/workspace/vinai/project/3d-ba-pc/data"
                                                 "/modelnet40_ply_hdf5_2048")
    points = x_train[12]
    points_attack = add_object_to_points(points=points, scale=0.4)
    pcd_attack = o3d.geometry.PointCloud()
    pcd_attack.points = o3d.utility.Vector3dVector(points_attack)
    pcd_attack.paint_uniform_color([0.5, 0.5, 0.5])
    num_point = points.shape[0]
    mask = np.concatenate([np.zeros((num_point, 1)), np.ones((128, 1))])
    new_points, index = farthest_point_sample_with_index(points_attack, npoint=1024)
    # new_points, index = random_sample_with_index(points_attack, npoint=1024)
    mask = mask[index, :]
    backdoor_point = int(np.sum(mask, axis=0))
    print("Backdoor Point Remain : {} points".format(backdoor_point))
    visualize_point_cloud_with_backdoor(points=new_points, mask=mask)

    # pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    # print("Paint the 1500th point red.")
    # pcd.colors[1500] = [1, 0, 0]
    # print("Find its 200 nearest neighbors, paint blue.")
    # [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[1500], 200)
    # np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
    # print("Find its neighbors with distance less than 0.2, paint green.")
    # [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[1500], 0.2)
    # np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]
    # print("Visualize the point cloud ")
