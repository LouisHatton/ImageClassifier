import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torchvision import datasets
from plotter import handle_predictions


def test(cnn, loaders, batch_size):
    # Test the model
    cnn.eval() 
    with torch.no_grad():
        totalImages = len(loaders) * batch_size
        all_accuracy = []
        # Iterate through the test set and calculate the accuracy.
        for i, (images, labels) in enumerate(loaders):
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            all_accuracy.append(accuracy)
            pass

    meanAccuracy = (sum(all_accuracy) / len(all_accuracy)) * 100
    print(f'Test Accuracy of the Model on {totalImages} test images: {meanAccuracy:0.2f}%')
    return meanAccuracy

def predict(cnn, loaders):
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    sample = next(iter(loaders))
    imgs, lbls = sample
    actual_class = lbls[:10].numpy()
    test_output, last_layer = cnn(imgs[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(f'Prediction Class: \t {pred_y}')
    print(f'Actual Class: \t \t {actual_class}')
    handle_predictions(imgs[:10], pred_y, actual_class, classes)


def testPlot(cnn, loaders):
    test_data = datasets.FashionMNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
    )

    figure = plt.figure(figsize=(10, 8))
    cols, rows = 5, 5
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(test_data), size=(1,)).item()
        img, label = test_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
        
    plt.show()