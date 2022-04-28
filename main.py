import torch
import time
import random
from utils import *
from models import *
from train import *
from test import *

# # # --- Configuration --- # # #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 100 
UPDATE_EVERY_X_BATCHES = 200
MODEL_NAME = "SingleNetworkTesting"


def saveModel(model, path):
    torch.save(model.state_dict(), path)
    print("Model saved as: ", path.split("./trained_models/")[1])

def loadModel(model, path):
    model.load_state_dict(torch.load(path))
    print("Model loaded from: ", path.split("./trained_models/")[1])

def train_model(model, train_loader, test_loader, num_epochs):
    tic = time.perf_counter()
    
    # Train the model
    train(num_epochs, model, train_loader, BATCH_SIZE, UPDATE_EVERY_X_BATCHES)

    toc = time.perf_counter()
    print(f"\n⌛ Model finished training in {toc - tic:0.2f} seconds")

    # Test the model
    print("\n🎯 Testing Accuracy... \n")
    mean_accuracy = test(model, test_loader, BATCH_SIZE)

    # Save the model
    print("\n💾 Saving the model... \n")
    rndNum = random.randint(0, 1000)
    saveMean = int(mean_accuracy * 100) 
    path = f"./trained_models/{MODEL_NAME}_{rndNum}_{saveMean}.pth"
    saveModel(model, path)

    # Predict the model
    print("\n🔮 Running Visual Predictions... \n")
    predict(model, test_loader)

def test_trained_model(model, test_loader, path, run_accuracy):
    print("\n🔃 Loading the model... \n")
    loadModel(model, path)

    if run_accuracy:
        print("\n🎯 Testing Accuracy... \n")
        mean_accuracy = test(model, test_loader, BATCH_SIZE)

    # Predict the model
    print("\n🔮 Running Visual Predictions... \n")
    predict(model, test_loader)



if (__name__ == "__main__"):    
    # Get the train and test data loaders
    print("\n🗃️  Loading the dataset... \n")
    train_loader, test_loader = getLoaders(batch_size=BATCH_SIZE)

    # Get the model
    model = SingleNetwork()
    
    train_model(
            model=model,
            train_loader=train_loader, 
            test_loader=test_loader,
            num_epochs=3
    )

    # trained_model_path = "./trained_models/SingleNetworkTesting_279_8920.pth"
    # test_trained_model(
    #                 model=model, 
    #                 test_loader=test_loader, 
    #                 path=trained_model_path, 
    #                 run_accuracy=False
    # )