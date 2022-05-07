from torch import dropout, max_pool2d
import torch.nn as nn
from torchsummary import summary


class FasterNet(nn.Module):
    def __init__(self):
        ch = 32
        super(FasterNet, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=3,      # Input is an RGB image
                out_channels=ch,    # Number of channels produced by the convolution
                kernel_size=3,      # 5x5 matrix which we slide over the image
                stride=1,           # The number of pixels to pass when sliding the kernel
                padding=1,          # To preserve the size of the image
            ),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=ch,
                out_channels=ch * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=ch * 2,
                out_channels=ch * 3,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(ch * 3),
            # nn.Conv2d(
            #     in_channels=96,
            #     out_channels=96,
            #     kernel_size=3,
            #     stride=1,
            #     padding=0,
            # ),
            nn.LeakyReLU(),
            
        )

        self.conv4 = nn.Sequential(
            nn.Dropout2d(p=0.5, inplace=True),
            nn.Conv2d(
                in_channels=ch * 3,
                out_channels=ch * 4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # nn.Conv2d(
            #     in_channels=128,
            #     out_channels=128,
            #     kernel_size=3,
            #     stride=1,
            #     padding=0,
            # ),
            nn.BatchNorm2d(ch*4),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=ch * 4,
                out_channels=ch * 5,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        nn.Dropout2d(p=0.5, inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                in_channels=ch * 5,
                out_channels=ch * 5,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.out = nn.Sequential(
            # nn.Linear(2048, 1024),
            nn.Linear((ch * 5) * (4 * 4), 10),
            # nn.Linear(1024, 10),
            # nn.Linear(1024, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization
    
if __name__ == '__main__':
    net = FasterNet()
    summary(net, (3, 32, 32))


