from torch import batch_norm
from torch import reshape
from torchsummary import summary
import torch.nn as nn

class ResidualNetwork(nn.Module):
    def __init__(self):
        # Double channels at each convolution,
        # similar to our "New Net" module.
        c = 32 # Number of pixels in

        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=c*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c*2),
            nn.LeakyReLU(inplace=True)
        )
        # c = 64
        c = c * 2 # Update c to the new number of channels we are handling
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c*2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # c = 128
        c = c * 2

        self.resi1 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1), 
            nn.BatchNorm2d(c), 
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c*2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # c = 256
        c = c * 2
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c*2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # c = 512
        c = c * 2

        self.resi2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1), 
            nn.BatchNorm2d(c), 
            nn.ReLU(inplace=True)
        )

        self.flatten = nn.Sequential(
            # nn.Dropout(p=0.02),
            nn.Linear(8192, 10),
        )

        self.remove_conv2_dimensions = nn.Sequential(
            nn.Conv2d(64,128,3,1),
            nn.Conv2d(128,128,3,1),
            # nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
        )

        self.remove_conv4_dimensions = nn.Sequential(
            nn.Conv2d(256,512,3,1),
            nn.Conv2d(512,512,3,1),
            nn.BatchNorm2d(512),
            nn.BatchNorm2d(512),
            # nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        con1 = self.conv1(x)
        con2 = self.conv2(con1)
        res1 = self.resi1(con2) + con2

        con3 = self.conv3(res1)
        con4 = self.conv4(con3)
        res2 = self.resi2(con4) + con4

        x = res2

        x = x.view(x.size(0), -1)       
        output = self.flatten(x)
        return output, x    # return x for visualization
    
if __name__ == '__main__':
    net = ResidualNetwork()
    summary(net, (3, 32, 32))
