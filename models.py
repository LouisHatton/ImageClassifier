import torch.nn as nn

class SingleNetwork(nn.Module):
    def __init__(self):
        super(SingleNetwork, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=3,      # Input is an RGB image
                out_channels=8,     # Number of channels produced by the convolution
                kernel_size=5,      # 5x5 matrix which we slide over the image
                stride=1,           # The number of pixels to pass when sliding the kernel
                padding=2,          # To preserve the size of the image
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization


