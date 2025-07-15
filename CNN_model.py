import torch
from torch import nn

from torch.utils.data import DataLoader
import torch.utils.data

from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FirstCNN(nn.Module):
    def __init__(self, input_shape=3):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=128 * 16 * 16,
                out_features=256
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=256,
                out_features=1
            )
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x