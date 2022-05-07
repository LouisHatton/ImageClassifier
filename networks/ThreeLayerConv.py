from torch import dropout, max_pool2d
import torch.nn as nn
from torchsummary import summary


class ThreeLayerConv(nn.Module):
    def __init__(self):
        ch = 16
        super(ThreeLayerConv, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Conv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.5),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=96,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # nn.Conv2d(
            #     in_channels=128,
            #     out_channels=128,
            #     kernel_size=3,
            #     stride=1,
            #     padding=1,
            # ),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=0.5),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.1),
        )

        self.out = nn.Sequential(
            # nn.Linear(2048, 1024),
            nn.Linear(32768, 10),
            # nn.Linear(1024, 10),
            # nn.Linear(1024, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization
    
if __name__ == '__main__':
    net = ThreeLayerConv()
    summary(net, (3, 32, 32))


