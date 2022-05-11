import torch
import time
import random
from networks.FasterNet import FasterNet
from networks.ResidualNetwork import ResidualNetwork
from networks.ThreeLayerConv import ThreeLayerConv
from plotter import *
from utils import *
from models import *
from train import *
from test import *

# # # --- Configuration --- # # #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 100
UPDATE_EVERY_X_BATCHES = 300
MODEL_NAME = "CIFAR10_TEST"


def saveModel(model, path):
    torch.save(model.state_dict(), path)
    print("Model saved to: ", path)

def loadModel(model, path):
    model.load_state_dict(torch.load(path))
    print("Model loaded from: ", path)

def train_model(model, train_loader, valid_loader, test_loader, num_epochs, with_TensorBoard):
    tic = time.perf_counter()
    
    # Train the model
    train(
        num_epochs=num_epochs,
        cnn=model,
        loaders=train_loader,
        valid_loader=valid_loader,
        batch_size=BATCH_SIZE, 
        update_every_x_batches=UPDATE_EVERY_X_BATCHES,
        with_TensorBoard=with_TensorBoard)

    toc = time.perf_counter()
    print(f"\nModel finished training in {toc - tic:0.2f} seconds")


    # Test the model
    print("\nTesting Accuracy... \n")
    mean_accuracy = test(model, test_loader, BATCH_SIZE)

    # Save the model
    print("\nSaving the model... \n")
    rndNum = random.randint(0, 1000)
    saveMean = int(mean_accuracy * 100) 
    path = f"./trained_models/{MODEL_NAME}_{rndNum}_{saveMean}.pth"
    saveModel(model, path)

    # Predict the model
    print("\nRunning Visual Predictions... \n")
    predict(model, test_loader)

def test_trained_model(model, test_loader, path, run_accuracy):
    print("\nLoading the model... \n")
    loadModel(model, path)

    if run_accuracy:
        print("\nTesting Accuracy... \n")
        mean_accuracy = test(model, test_loader, BATCH_SIZE)

    # Predict the model
    print("\nRunning Visual Predictions... \n")
    predict(model, test_loader)



if (__name__ == "__main__"):    
    # Get the train and test data loaders
    print("\nLoading the dataset... \n")
    train_loader, test_loader, validation_loader = getLoaders(batch_size=BATCH_SIZE)

    # Get the model
    model = ResidualNetwork()
    model.to(device)
    
    # Uncomment out the following line to train the model

    # train_model(
    #         model=model,
    #         train_loader=train_loader,
    #         valid_loader=validation_loader, 
    #         test_loader=test_loader,
    #         num_epochs=30,
    #         with_TensorBoard=True
    # )

    # Uncomment out the following line to test the model

    trained_model_path = "./CIFAR10_temp_accuracy_save.pth"
    test_trained_model(
                    model=model, 
                    test_loader=test_loader, 
                    path=trained_model_path, 
                    run_accuracy=True
    )
