import torch
from load_data import load_data


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    x = torch.from_numpy(x_train[0])
    x = x.unsqueeze(dim=0)
    x = x.transpose(2, 1)
    print(x.shape)
    print(knn(x, 20).shape)
