import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as grad
from config import *

from models.pointnet_base import PointNetBase


# -----------------------------------------------------------------------------
# Class for PointNetClassifier. Subclasses PyTorch's own "nn" module
#
# Computes the local embeddings and global features for an input set of points
##
class PointNetClassifier(nn.Module):

    def __init__(self, device, num_points=NUM_POINTS, K=3):
        # Call the super constructor
        super(PointNetClassifier, self).__init__()

        # Local and global feature extractor for PointNet
        self.base = PointNetBase(device=device)

        # Classifier for ShapeNet
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, 40)
        )

    # Take as input a B x K x N matrix of B batches of N points with K
    # dimensions
    def forward(self, x):
        # Only need to keep the global feature descriptors for classification
        # Output should be B x 1024
        x, _, T2 = self.base(x)

        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)

        # Returns a B x 40
        return x, T2


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = PointNetClassifier(num_points=NUM_POINTS, device=device)
    x = torch.randn([32, 3, 2048])
    out, _ = model(x)
