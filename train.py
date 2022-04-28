from torch.autograd import Variable
from torch import optim
import torch.nn as nn

def train(num_epochs, cnn, loaders, batch_size, update_every_x_batches):
    optimizer = optim.Adam(cnn.parameters(), lr = 0.01)   
    loss_func = nn.CrossEntropyLoss()

    cnn.train() # set model to training mode
        
    total_step = len(loaders)
    print("Total number of Images: ", total_step * batch_size)
    print("Total number of Epochs: ", num_epochs)
    print("Total number of Batches per Epoch: ", total_step)
    print("Total number of Images per Batch: ", batch_size)
    print("\n")
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders):
            b_img = Variable(images)   # batch images
            b_lbs = Variable(labels)   # batch labels

            # Forward pass though network
            output = cnn(b_img)[0]

            # Compute loss
            loss = loss_func(output, b_lbs)
            
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
        
        pass
    
    
    pass