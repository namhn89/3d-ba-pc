import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from load_data import get_data

matplotlib.use('Agg')


def scale_plot():
    # plt.axis("scaled")
    plt.gca().set_xlim(-1, 1)
    plt.gca().set_ylim(-1, 1)
    plt.gca().set_zlim(-1, 1)
    plt.gca().view_init(0, 0)


def draw_point_cloud(points):
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    plt.figure(figsize=(7, 7))
    plt.subplot(111, projection="3d")
    plt.gca().scatter(xs, ys, zs, zdir="y", s=5)
    scale_plot()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.axis('off')
    plt.savefig('a.jpg')


def pyplot_draw_point_cloud(points, output_filename):
    """ points is a Nx3 numpy array """
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure()
    xmin, xmax = -1 - 0.1, 1 + 0.1
    ymin, ymax = -1 - 0.1, 1 + 0.1
    zmin, zmax = -1 - 0.1, 1 + 0.1

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 1], points[:, 2], points[:, 0], s=1, c='black')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_zlabel('x')
    ax.set_xlim(ymin, ymax)
    ax.set_ylim(zmin, zmax)
    ax.set_zlim(xmin, xmax)
    plt.axis('off')
    plt.savefig(output_filename + '_x.jpg')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 2], points[:, 0], points[:, 1], s=1, c='black')
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(xmin, xmax)
    ax.set_zlim(ymin, ymax)
    plt.axis('off')
    plt.savefig(output_filename + '_y.jpg')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='black')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    plt.axis('off')
    plt.savefig(output_filename + '_z.jpg')
    plt.close()


def pyplot_draw_saliency_map(points, color, output_filename):
    """ points is a Nx3 numpy array """
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    xmin, xmax = np.min(points[:, 0]) - 0.1, np.max(points[:, 0]) + 0.1
    ymin, ymax = np.min(points[:, 1]) - 0.1, np.max(points[:, 1]) + 0.1
    zmin, zmax = np.min(points[:, 2]) - 0.1, np.max(points[:, 2]) + 0.1
    ax = fig.add_subplot(111, projection='3d')
    bar = ax.scatter(points[:, 2], points[:, 0], points[:, 1], c=color, cmap='hsv')
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(xmin, xmax)
    ax.set_zlim(ymin, ymax)
    plt.axis('off')
    cb = plt.colorbar(bar)
    cb.set_label('Score of saliency map')
    plt.savefig(output_filename)
    plt.close()


def pyplot_draw_point_cloud_natural(points, mask, output_filename):
    """ points is a Nx3 numpy array """
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    # xmin, xmax = np.min(points[:, 0]) - 0.1, np.max(points[:, 0]) + 0.1
    # ymin, ymax = np.min(points[:, 1]) - 0.1, np.max(points[:, 1]) + 0.1
    # zmin, zmax = np.min(points[:, 2]) - 0.1, np.max(points[:, 2]) + 0.1
    xmin, xmax = -1 - 0.1, 1 + 0.1
    ymin, ymax = -1 - 0.1, 1 + 0.1
    zmin, zmax = -1 - 0.1, 1 + 0.1

    ax = fig.add_subplot(111, projection='3d')
    for idx, color in enumerate(mask):
        if color[0] == 1.:
            ax.scatter(points[idx, 1], points[idx, 2], points[idx, 0], s=5, c='red')
        elif color[0] == 2.:
            ax.scatter(points[idx, 1], points[idx, 2], points[idx, 0], s=5, c='navy')
        elif color[0] == 3.:
            ax.scatter(points[idx, 1], points[idx, 2], points[idx, 0], s=5, c='magenta')
        else:
            ax.scatter(points[idx, 1], points[idx, 2], points[idx, 0], s=1, c='black')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_zlabel('x')
    ax.set_xlim(ymin, ymax)
    ax.set_ylim(zmin, zmax)
    ax.set_zlim(xmin, xmax)
    plt.axis('off')
    plt.savefig(output_filename + '_x.jpg')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points[:, 2], points[:, 0], points[:, 1], s=5, c='b')
    for idx, color in enumerate(mask):
        if color[0] == 1.:
            ax.scatter(points[idx, 2], points[idx, 0], points[idx, 1], s=5, c='red')
        elif color[0] == 2.:
            ax.scatter(points[idx, 2], points[idx, 0], points[idx, 1], s=5, c='navy')
        elif color[0] == 3.:
            ax.scatter(points[idx, 2], points[idx, 0], points[idx, 1], s=5, c='magenta')
        else:
            ax.scatter(points[idx, 2], points[idx, 0], points[idx, 1], s=1, c='black')
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(xmin, xmax)
    ax.set_zlim(ymin, ymax)
    plt.axis('off')
    plt.savefig(output_filename + '_y.jpg')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, c='b')
    for idx, color in enumerate(mask):
        if color[0] == 1.:
            ax.scatter(points[idx, 0], points[idx, 1], points[idx, 2], s=5, c='red')
        elif color[0] == 2.:
            ax.scatter(points[idx, 0], points[idx, 1], points[idx, 2], s=5, c='navy')
        elif color[0] == 3.:
            ax.scatter(points[idx, 0], points[idx, 1], points[idx, 2], s=5, c='magenta')
        else:
            ax.scatter(points[idx, 0], points[idx, 1], points[idx, 2], s=1, c='black')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    plt.axis('off')
    plt.savefig(output_filename + '_z.jpg')
    plt.close()


def pyplot_draw_point_cloud_nat_and_adv(points, points_adv, output_filename):
    """ points is a Nx3 numpy array """
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    xmin, xmax = np.min(points[:, 0]) - 0.1, np.max(points[:, 0]) + 0.1
    ymin, ymax = np.min(points[:, 1]) - 0.1, np.max(points[:, 1]) + 0.1
    zmin, zmax = np.min(points[:, 2]) - 0.1, np.max(points[:, 2]) + 0.1

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 1], points[:, 2], points[:, 0], s=5, c='b')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_zlabel('x')
    ax.set_xlim(ymin, ymax)
    ax.set_ylim(zmin, zmax)
    ax.set_zlim(xmin, xmax)
    plt.axis('off')
    plt.savefig(output_filename + '_1nat_x.jpg')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_adv[:, 1], points_adv[:, 2], points_adv[:, 0], s=5, c='r')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_zlabel('x')
    ax.set_xlim(ymin, ymax)
    ax.set_ylim(zmin, zmax)
    ax.set_zlim(xmin, xmax)
    plt.axis('off')
    plt.savefig(output_filename + '_2adv_x.jpg')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 2], points[:, 0], points[:, 1], s=5, c='b')
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(xmin, xmax)
    ax.set_zlim(ymin, ymax)
    plt.axis('off')
    plt.savefig(output_filename + '_3nat_y.jpg')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_adv[:, 2], points_adv[:, 0], points_adv[:, 1], s=5, c='r')
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(xmin, xmax)
    ax.set_zlim(ymin, ymax)
    plt.axis('off')
    plt.savefig(output_filename + '_4adv_y.jpg')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, c='b')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    plt.axis('off')
    plt.savefig(output_filename + '_5nat_z.jpg')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_adv[:, 0], points_adv[:, 1], points_adv[:, 2], s=5, c='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    plt.axis('off')
    plt.savefig(output_filename + '_6adv_z.jpg')
    plt.close()


def plot_nat_interval_adv(points, drop_points, image_filename):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xmin, xmax = np.min(points[:, 0]) - 0.1, np.max(points[:, 0]) + 0.1
    ymin, ymax = np.min(points[:, 1]) - 0.1, np.max(points[:, 1]) + 0.1
    zmin, zmax = np.min(points[:, 2]) - 0.1, np.max(points[:, 2]) + 0.1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 2], points[:, 0], points[:, 1], s=1, c='k')
    ax.scatter(drop_points[:, 2], drop_points[:, 0], drop_points[:, 1], s=1, c='k')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_zlabel('x')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(xmin, xmax)
    ax.set_zlim(ymin, ymax)
    plt.axis('off')
    plt.savefig(image_filename + '_y1.jpg', bbox_inches='tight')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 2], points[:, 0], points[:, 1], s=1, c='k')
    ax.scatter(drop_points[:, 2], drop_points[:, 0], drop_points[:, 1], s=5, c='r')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_zlabel('x')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(xmin, xmax)
    ax.set_zlim(ymin, ymax)
    plt.axis('off')
    plt.savefig(image_filename + '_y2.jpg', bbox_inches='tight')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 2], points[:, 0], points[:, 1], s=1, c='k')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_zlabel('x')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(xmin, xmax)
    ax.set_zlim(ymin, ymax)
    plt.axis('off')
    plt.savefig(image_filename + '_y3.jpg', bbox_inches='tight')
    plt.close()


def write_ply_color(points, labels, out_filename, num_classes=None):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    import matplotlib.pyplot as pyplot
    labels = labels.astype(int)
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels) + 1
    else:
        assert (num_classes > np.max(labels))
    fout = open(out_filename, 'w')
    # colors = [pyplot.cm.hsv(i/float(num_classes)) for i in range(num_classes)]
    colors = [pyplot.cm.jet(i / float(num_classes)) for i in range(num_classes)]
    for i in range(N):
        c = colors[labels[i]]
        c = [int(x * 255) for x in c]
        fout.write('v %f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()


if __name__ == '__main__':
    x_train, _, _, _, _ = get_data("modelnet40")
    print(x_train.shape)
