from pykeops.torch import generic_argkmin
import torch.nn as nn
import torch


def knn3(K=20, D=3):
    knn = generic_argkmin(
        'SqDist(x, y)',
        'a = Vi({})'.format(K),
        'x = Vi({})'.format(D),
        'y = Vj({})'.format(D),
    )
    return knn


class RandLANet(nn.Module):
    def __init__(self, in_c, out_c, kernals, num_neighbor=20, bias=True, is_dim9=False):  # ,device=None):
        super(RandLANet, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.num_neighbor = num_neighbor
        self.kernel_size = kernals.shape[1]
        self.mlp = nn.Conv1d(7, in_c, kernel_size=1, bias=bias)
        self.mlp_weight = nn.Conv1d(in_c * 2, in_c * 2, kernel_size=1, bias=False)
        self.conv = nn.Linear(2 * in_c, out_c, bias=bias)
        self.mlp_out = nn.Conv1d(in_c, out_c, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=1)  # Sparsemax(dim=-1) #
        self.bn = nn.BatchNorm1d(out_c)
        self.knn = knn3(num_neighbor)

    def forward(self, x, feature):
        bsize, num_feat, num_pts = feature.size()
        x = x.permute(0, 2, 1).contiguous()
        neigh_index = self.knn(x, x)
        x = x.view(bsize * num_pts, 3)

        idx_base = torch.arange(0, bsize, device=x.device).view(-1, 1, 1) * num_pts
        neigh_index = (neigh_index + idx_base).view(-1)  # bsize * num_pts * num_neighbor

        #### relative position ####
        x_neighs = x[neigh_index, :].view(bsize * num_pts, self.num_neighbor, 3)
        x_repeat = x_neighs[:, 0:1, :].expand_as(x_neighs)
        x_relative = x_neighs - x_repeat
        x_dis = torch.norm(x_relative, dim=-1, keepdim=True)
        x_feats = torch.cat([x_repeat, x_relative, x_dis], dim=-1)
        x_feats = self.mlp(x_feats.permute(0, 2, 1).contiguous())

        feats = feature.permute(0, 2, 1).contiguous().view(bsize * num_pts, num_feat)
        feats = feats[neigh_index, :].view(bsize * num_pts, self.num_neighbor, num_feat)
        feats = feats.permute(0, 2, 1).contiguous()
        feats = torch.cat([feats, x_feats], dim=1)

        feats = torch.sum(self.softmax(self.mlp_weight(feats)) * feats, dim=-1)
        feats = feats.view(bsize * num_pts, 2 * num_feat)
        out_feat = self.conv(feats).view(bsize, num_pts, self.out_c)

        out_feat = out_feat.permute(0, 2, 1).contiguous() + self.mlp_out(feature)
        return self.bn(out_feat)


if __name__ == '__main__':
    x = torch.randn(3, )
