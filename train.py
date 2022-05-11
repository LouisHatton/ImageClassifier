from audioop import avg
from operator import itemgetter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim
import torch
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

def train(num_epochs, cnn, loaders, valid_loader, batch_size, update_every_x_batches, with_TensorBoard=False):
    
    optimizer = optim.Adam(cnn.parameters(), lr = 0.002)   
    loss_func = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=0, verbose=True)

    cnn.train() # set model to training mode
        
    total_step = len(loaders)
    print("Total number of Images: ", total_step * batch_size)
    print("Total number of Epochs: ", num_epochs)
    print("Total number of Batches per Epoch: ", total_step)
    print("Total number of Images per Batch: ", batch_size)
    print("\n")

    if with_TensorBoard: writer = SummaryWriter()
    if with_TensorBoard: writer.add_scalar("Accuracy/train", 0, 0)
    if with_TensorBoard: writer.add_scalar("Accuracy/validation", 0, 0)


    iteration = 0
    min_valid_loss = 100
    max_valid_acc = 0
    stop_flag = 0
    max_flags = 3
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        total_loss = 0
        data = enumerate(loaders)
        cnn.train()
        for i, (images, labels) in data:
            b_img = Variable(images)   # batch images
            b_lbs = Variable(labels)   # batch labels

            # Forward pass though networks
            output = cnn(b_img)[0]

            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            

            # Compute loss
            loss = loss_func(output, b_lbs)
            total_loss += loss.item()
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
        avg_loss = total_loss / iteration
        if with_TensorBoard: writer.add_scalar("Accuracy/train", accuracy, epoch + 1)
        if with_TensorBoard: writer.add_scalar("AvgLoss/train", avg_loss, epoch + 1)
        print(f"\nEpoch {epoch+1}, Train Average Loss: {avg_loss:0.4f}")
        print(f"Epoch {epoch+1}, Train Accuracy: {accuracy:0.2f}%\n")

        # Calculate the Validation Accuracy of the model
        cnn.eval()
        iteration = 0
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_loader):
                b_img = Variable(images)
                b_lbs = Variable(labels)
                output = cnn(b_img)[0]
                _, predicted = output.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                loss = loss_func(output, b_lbs)
                if with_TensorBoard: writer.add_scalar("Loss/validation", loss, iteration)
                total_loss += loss.item()
                iteration += 1
        
        accuracy = 100 * (correct / total)
        avg_loss = total_loss / iteration
        if with_TensorBoard: writer.add_scalar("Accuracy/validation", accuracy, epoch + 1)
        if with_TensorBoard: writer.add_scalar("AvgLoss/validation", avg_loss, epoch + 1)
        print(f"Epoch {epoch+1}, Validation Average Loss: {avg_loss:0.4f} ")
        print(f"Epoch {epoch+1}, Validation Accuracy: {accuracy:0.2f}%\n")

        # Reduces training rate if val loss plateaus 
        scheduler.step(avg_loss)

        if (avg_loss < min_valid_loss):
            min_valid_loss = avg_loss
            stop_flag = 0
            print("Loss Improved - saving temp loss file...\n")
            PATH = './CIFAR10_temp_loss_save.pth'
            torch.save(cnn.state_dict(), PATH)
        elif ((((avg_loss - min_valid_loss) / min_valid_loss) * 100 < 10) & (stop_flag+1 == max_flags)):
            print(f"Loss Failed to improve - Not greater than 10% - Give BOD\n")
        else:
            stop_flag += 1
            print(f"Loss Failed to improve - Attempt: {stop_flag}/{max_flags}\n")
            if stop_flag == max_flags:
                print("\nEarly Stopping Criteria Met\n")
                break
        
        if (max_valid_acc < accuracy):
            max_valid_acc = accuracy
            print("Accuracy Improved - saving temp accuracy file...\n")
            PATH = './CIFAR10_temp_accuracy_save.pth'
            torch.save(cnn.state_dict(), PATH)

    
    if with_TensorBoard: writer.flush()