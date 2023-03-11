import torch.nn as nn


class RefineNetwork(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, mid_channels=128):
        super(RefineNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        return self.model(x)
