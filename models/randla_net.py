import time

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_points import knn
except (ModuleNotFoundError, ImportError):
    from torch_points_kernels import knn


class SharedMLP(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            transpose=False,
            padding_mode='zeros',
            bn=False,
            activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
            Forward pass of the network

            Parameters
            ----------
            input: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors, device):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())

        self.device = device

    def forward(self, coords, features, knn_output):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d, N, 1)
                features of the point cloud
            neighbors: tuple

            Returns
            -------
            torch.Tensor, shape (B, 2*d, N, K)
        """
        # finding neighboring points
        idx, dist = knn_output
        B, N, K = idx.size()
        # idx(B, N, K), coords(B, N, 3)
        # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = extended_coords[b, i, extended_idx[b, i, n, k], k]
        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K)
        neighbors = torch.gather(extended_coords, 2, extended_idx)  # shape (B, 3, N, K)
        # if USE_CUDA:
        #     neighbors = neighbors.cuda()

        # relative point position encoding
        concat = torch.cat((
            extended_coords,
            neighbors,
            extended_coords - neighbors,
            dist.unsqueeze(-3)
        ), dim=-3).to(self.device)
        return torch.cat((
            self.mlp(concat),
            features.expand(B, -1, N, K)
        ), dim=-3)


class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2)
        )
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())

    def forward(self, x):
        r"""
            Forward pass

            Parameters
            ----------
            x: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        scores = self.score_fn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # sum over the neighbors
        features = torch.sum(scores * x, dim=-1, keepdim=True)  # shape (B, d_in, N, 1)

        return self.mlp(features)


class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors, device):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(d_in, d_out // 2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2 * d_out)
        self.shortcut = SharedMLP(d_in, 2 * d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(d_out // 2, num_neighbors, device)
        self.lse2 = LocalSpatialEncoding(d_out // 2, num_neighbors, device)

        self.pool1 = AttentivePooling(d_out, d_out // 2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, features):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d_in, N, 1)
                features of the point cloud

            Returns
            -------
            torch.Tensor, shape (B, 2*d_out, N, 1)
        """
        knn_output = knn(coords.cpu().contiguous(), coords.cpu().contiguous(), self.num_neighbors)

        x = self.mlp1(features)

        x = self.lse1(coords, x, knn_output)
        x = self.pool1(x)

        x = self.lse2(coords, x, knn_output)
        x = self.pool2(x)

        return self.lrelu(self.mlp2(x) + self.shortcut(features))


class RandLANet(nn.Module):
    def __init__(self, d_in, num_classes, num_neighbors=16, decimation=4, device=torch.device('cpu')):
        super(RandLANet, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        self.fc_start = nn.Linear(d_in, 8)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(8, 16, num_neighbors, device),
            LocalFeatureAggregation(32, 64, num_neighbors, device),
            LocalFeatureAggregation(128, 128, num_neighbors, device),
            LocalFeatureAggregation(256, 256, num_neighbors, device),
            LocalFeatureAggregation(512, 512, 8, device),
            # LocalFeatureAggregation(256, 256, 8, device)
            # LocalFeatureAggregation(512, 512, num_neighbors, device),
        ])

        self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())

        # decoding layers
        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.ReLU()
        )
        self.decoder = nn.ModuleList([
            SharedMLP(1024, 256, **decoder_kwargs),
            SharedMLP(512, 128, **decoder_kwargs),
            SharedMLP(256, 32, **decoder_kwargs),
            SharedMLP(64, 8, **decoder_kwargs)
        ])

        # final semantic prediction
        self.fc_end = nn.Sequential(
            SharedMLP(8, 64, bn=True, activation_fn=nn.ReLU()),
            SharedMLP(64, 32, bn=True, activation_fn=nn.ReLU()),
            nn.Dropout(),
            SharedMLP(32, num_classes)
        )

        self.fc1 = nn.Linear(2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(256, num_classes)
        self.device = device

        self = self.to(device)

    def forward(self, input):
        r"""
            Forward pass

            Parameters
            ----------
            input: torch.Tensor, shape (B, N, d_in)
                input points

            Returns
            -------
            torch.Tensor, shape (B, num_classes, N)
                segmentation scores for each point
        """
        N = input.size(1)
        d = self.decimation

        coords = input[..., :3].clone().cpu()
        x = self.fc_start(input).transpose(-2, -1).unsqueeze(-1)
        x = self.bn_start(x)  # shape (B, d, N, 1)

        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        x_stack = []

        permutation = torch.randperm(N)
        coords = coords[:, permutation]
        x = x[:, :, permutation]

        for lfa in self.encoder:
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            x = lfa(coords[:, :N // decimation_ratio], x)
            x_stack.append(x.clone())
            decimation_ratio *= d
            # print(decimation_ratio)
            # print(x.shape)
            x = x[:, :, :N // decimation_ratio]

        # # >>>>>>>>>> ENCODER
        # print(x.shape)
        # x = self.mlp(x)
        x = torch.squeeze(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = torch.flatten(x, start_dim=1)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.drop3(x)
        x = self.fc4(x)
        x = F.log_softmax(x, dim=1)
        print(x.shape)

        # # <<<<<<<<<< DECODER
        # for mlp in self.decoder:
        #     neighbors, _ = knn(
        #         coords[:, :N // decimation_ratio].cpu().contiguous(),  # original set
        #         coords[:, :d * N // decimation_ratio].cpu().contiguous(),  # upsampled set
        #         1
        #     )  # shape (B, N, 1)
        #     neighbors = neighbors.to(self.device)
        #
        #     extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)
        #
        #     x_neighbors = torch.gather(x, -2, extended_neighbors)
        #
        #     x = torch.cat((x_neighbors, x_stack.pop()), dim=1)
        #
        #     x = mlp(x)
        #
        #     decimation_ratio //= d
        # print(x.shape)
        #
        # # >>>>>>>>>> DECODER
        # # inverse permutation
        # x = x[:, :, torch.argsort(permutation)]
        # print(x.shape)

        # scores = self.fc_end(x)

        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss


if __name__ == '__main__':
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    d_in = 3
    cloud = torch.randn(32, 2048, d_in).to(device)
    model = RandLANet(d_in, 40, 16, 4, device)
    print(model)
    model.eval()

    t0 = time.time()
    pred = model(cloud)
    print(pred.shape)
    t1 = time.time()
    # print(pred)
    print(t1 - t0)
