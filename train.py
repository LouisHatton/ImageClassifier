from operator import itemgetter
from torch.autograd import Variable
from torch import optim
import torch
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

def train(num_epochs, cnn, loaders, test_loaders, batch_size, update_every_x_batches, with_TensorBoard=False):
    
    optimizer = optim.Adam(cnn.parameters(), lr = 0.01)   
    loss_func = nn.CrossEntropyLoss()

    cnn.train() # set model to training mode
        
    total_step = len(loaders)
    print("Total number of Images: ", total_step * batch_size)
    print("Total number of Epochs: ", num_epochs)
    print("Total number of Batches per Epoch: ", total_step)
    print("Total number of Images per Batch: ", batch_size)
    print("\n")

    if with_TensorBoard: writer = SummaryWriter()

    iteration = 0
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        data = enumerate(loaders)
        for i, (images, labels) in data:
            b_img = Variable(images)   # batch images
            b_lbs = Variable(labels)   # batch labels

            # Forward pass though network
            output = cnn(b_img)[0]

            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            

            # Compute loss
            loss = loss_func(output, b_lbs)
            if with_TensorBoard: writer.add_scalar("Loss/train", loss, iteration)
            iteration += 1
            
            # clear gradients
            optimizer.zero_grad()
            
            # backpropagation - compute the gradients 
            loss.backward()

            # apply the gradients
            optimizer.step()
            
            if (i+1) % update_every_x_batches == 0:
                print ('Epoch [{}/{}], \t Batch [{}/{}], \t Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                pass

        accuracy = 100 * (correct / total)
        if with_TensorBoard: writer.add_scalar("Accuracy/train", accuracy, epoch + 1)
        print(f"\nEpoch {epoch+1}, Train Accuracy: {accuracy:0.2f}%\n ")
    
    if with_TensorBoard: writer.flush()