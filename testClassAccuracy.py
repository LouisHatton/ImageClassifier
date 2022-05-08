import torch
from main import BATCH_SIZE
from networks.FasterNet import FasterNet
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ]
)

def main():
    model = FasterNet()
    PATH = './cifar_model_faster_net_6649_16_batches_002_loss.pth'
    BATCH_SIZE = 16
    model.load_state_dict(torch.load(PATH))
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=True)

    classes = test_dataset.classes

    trained_model = model

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = trained_model(images)[0]
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    print('\nAccuracy of %5s : %2d %%' % (
            "total", 100 * sum(class_correct) / sum(class_total)))

if (__name__ == "__main__"):
    main()