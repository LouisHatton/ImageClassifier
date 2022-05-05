import torch.nn as nn

class SingleNetwork(nn.Module):
    def __init__(self):
        super(SingleNetwork, self).__init__()
        self.convSeq1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=3,      # Input is an RGB image
                out_channels=12,     # Number of channels produced by the convolution
                kernel_size=5,      # 5x5 matrix which we slide over the image
                stride=1,           # The number of pixels to pass when sliding the kernel
                padding=0           # -4 to output dimensionality (28x28)
            ),           
            nn.BatchNorm2d(12),                  
            nn.LeakyReLU(),
            #nn.MaxPool2d(kernel_size=2) #/2 dimensionality (14x14)
        )

        self.convSeq2 = nn.Sequential(
            nn.Conv2d(
                in_channels=12,     # In channels = out channels of last convolution
                out_channels=60,    # Number of channels produced by the convolution
                kernel_size=5,      # 5x5 matrix which we slide over the image
                stride=1,           # The number of pixels to pass when sliding the kernel
                padding=0           # -4 to dimensions (24x24)
                ),
                nn.BatchNorm2d(60),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2)    #Halves dimensions (12x12)
        )

        self.convSeq3 = nn.Sequential(
            nn.Conv2d(
                in_channels=60,     # In channels = out channels of last convolution
                out_channels=80,    # Number of channels produced by the convolution
                kernel_size=3,      # 5x5 matrix which we slide over the image
                stride=1,           # The number of pixels to pass when sliding the kernel
                padding=0           # -2 dimensionality  (10x10)
                ),
                nn.BatchNorm2d(80),
                nn.LeakyReLU(),
                #nn.MaxPool2d(kernel_size=2)    #Halves dimensions (5x5)
        )

        self.convSeq4 = nn.Sequential(
            nn.Conv2d(
                in_channels=80,     # In channels = out channels of last convolution
                out_channels=120,    # Number of channels produced by the convolution
                kernel_size=3,      # 5x5 matrix which we slide over the image
                stride=1,           # The number of pixels to pass when sliding the kernel
                padding=0           # -2 dimensionality  (8x8)
                ),
                nn.BatchNorm2d(120),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2)    #Halves dimensions (4x4)
        )

        self.droprate = 0.1
        self.drop = nn.Dropout(p=self.droprate,inplace=True)
        # fully connected layer, output 10 classes
        self.out = nn.Linear(1920, 10) #EDIT firstnum = no. output of last layer * their dimensions (15x15)

    def forward(self, x):
        x = self.convSeq1(x)
        x = self.convSeq2(x)
        '''
        if self.training == True:
            x=self.drop(x)
        else:
            x *= 1-self.droprate
        self.droprate = 0.2
        '''
        x = self.convSeq3(x)
        '''
        if self.training == True:
            x=self.drop(x)
        else:
            x *= 1-self.droprate
        self.droprate = 0.1
        '''

        x = self.convSeq4(x)
        x = x.view(x.size(0), -1)   
        output = self.out(x)
        return output, x    # return x for visualization


