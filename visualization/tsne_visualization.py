import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from load_data import load_data


def calculate_tsne(data):
    time_start = time.time()
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=100)
    tsne_results = tsne.fit_transform(data)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    return tsne_results


def calculate_pca(data):
    time_start = time.time()
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(data)
    print('PCA done! Time elapsed: {} seconds'.format(time.time() - time_start))
    return pca_results


def save_image_from_tsne(data, label, name_file=None):
    tsne = calculate_tsne(data)
    data_label = {
        "label": label
    }
    data_frame = pd.DataFrame(data_label, columns=["label"])
    data_frame["x"] = tsne[:, 0]
    data_frame["y"] = tsne[:, 1]

    # print(data_frame)
    # print(data_frame.y.unique().shape[0])

    plt.figure(figsize=(10, 10))
    sns_plot = sns.scatterplot(
        x="x", y="y",
        hue="label",
        palette=sns.color_palette("hls", data_frame.label.unique().shape[0]),
        data=data_frame,
        legend="full",
        alpha=0.3,
    )
    fig = sns_plot.get_figure()
    if name_file is not None:
        fig.savefig(name_file)
    else:
        plt.show()


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_data(
        '/home/nam/workspace/vinai/project/3d-ba-pc/data/modelnet40_ply_hdf5_2048')

    data = []
    for point in X_train[0:1000]:
        point = point[:1024, :]
        print(point.shape)
        point = point.reshape(-1, 3 * 1024)
        data.append(point)

    data = np.concatenate(data)
    label = np.squeeze(Y_train[0:1000])
    save_image_from_tsne(data, label, name_file='../test.png')
