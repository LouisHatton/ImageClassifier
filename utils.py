from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


train_data = datasets.CIFAR10(
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True, ## If you do not have the data set stored locally at '/data' set to True
)
test_data = datasets.CIFAR10(
    root = 'data',
    train = True,
    transform = ToTensor()
)

def visualizeDATA():
    figure = plt.figure(figsize=(10, 8))
    cols, rows = 5, 5
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        img, label = train_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")

    plt.show()

def getLoaders(batch_size=100):
    return (
    torch.utils.data.DataLoader(train_data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=1),
    
    torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=1),
    )