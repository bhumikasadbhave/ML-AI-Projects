"""
Created: Bhumika Sadbhave, 7/6/2024
"""

import torch, torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.optim as optim
import matplotlib.pyplot as plt


def mlp_train(model, model_type, trainset, testset, in_features=784, n_epochs=10, batch_size=32, optimizer_param='SGD', optimizer_lr=0.001, momentum=0, device=torch.device('cpu')):
    """Train model for classification of FashionMNIST data

    Args:
    model: Un-trained model 
    n_epochs: Number of epochs for training (int)
    batch_size: Batch Size for training (int)
    optimizer_lr: Learning Rate to use for Optimizer (float)

    Returns:
    model: Model trained and evaluated on FashionMNIST data
    loss: tuple of average train loss and average test loss(train_loss, test_loss)
    accuracy: tuple of average train accuracy and average test accuracy(train_loss, test_loss)
    """

    #Load Dataset
    trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = True)
    testloader = DataLoader(testset, batch_size = batch_size)

    #Optimizer and Loss Function
    loss_function = nn.CrossEntropyLoss()
    if optimizer_param == 'SGD':
        optimizer = optim.SGD(params = model.parameters(), lr = optimizer_lr, momentum=momentum)
    elif optimizer_param == 'Adam':
        optimizer = optim.Adam(params = model.parameters(), lr = optimizer_lr)
    elif optimizer_param == 'RMSProp':
        optimizer = optim.RMSprop(params = model.parameters(), lr = optimizer_lr)
    
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    #Training and Testing
    train_losses = []
    train_accuracy = []
    test_losses = []
    test_accuracy = []

    for epoch in range(n_epochs):
        batch_losses = []
        batch_accuracy = []
        #for x_batch, y_batch in trainloader:
        for x_batchcpu, y_batchcpu in trainloader:
            
            x_batch = x_batchcpu.to(device)
            y_batch = y_batchcpu.to(device)
            model.train()
            
            #Forward pass
            if model_type == 'ANN':
                x_batch = x_batch.view(x_batch.size(0), -1)
            logits = model(x_batch)

            #Loss function and Accuracy
            loss = loss_function(logits, y_batch)
            batch_losses.append(loss.item())

            _,preds = torch.max(logits,1)   
            accuracy = (torch.sum(preds==y_batch)/len(preds))*100  
            batch_accuracy.append(accuracy.item())

            #Backward pass - get gradients
            loss.backward()

            #Grad Step
            optimizer.step()

            #Zero out the gradients
            optimizer.zero_grad()

        #scheduler.step()
        avg_loss = sum(batch_losses)/len(batch_losses)
        avg_accuracy = sum(batch_accuracy)/len(batch_accuracy)
        train_losses.append(avg_loss)
        train_accuracy.append(avg_accuracy)

        #Monitoring on Test
        with torch.no_grad():
            model.eval()
            test_batch_losses = []
            test_batch_accuracy = []               
            #for x_testbatch, y_testbatch in testloader:
            for xt_batchcpu, yt_batchcpu in testloader:
                
                x_testbatch = xt_batchcpu.to(device)
                y_testbatch = yt_batchcpu.to(device)
                
                if model_type == 'ANN':
                    x_testbatch = x_testbatch.view(x_testbatch.size(0), -1)
                logits = model(x_testbatch)
                
                loss = loss_function(logits, y_testbatch)
                _,preds = torch.max(logits,1)
                accuracy = (torch.sum(preds == y_testbatch)/len(preds))*100                    
                test_batch_losses.append(loss.item())
                test_batch_accuracy.append(accuracy.item())
                
            avg_test_loss = sum(test_batch_losses)/len(test_batch_losses)
            avg_test_accuracy = sum(test_batch_accuracy)/len(test_batch_accuracy)
            test_losses.append(avg_test_loss)
            test_accuracy.append(avg_test_accuracy)
        
        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}')
            
    return model, train_losses, test_losses, train_accuracy, test_accuracy



def mlp_apply(model, model_type, testset, indices):
    """
    Uses the trained model to classify the examples specified by the indices in the test set.
    Displays the images in a grid together with their true and predicted labels.

    Args:
    model: Trained model
    testset: Test dataset
    indices: List of indices specifying which examples to extract and classify
    """
    
    model.eval()
    
    #Labels Dictionary
    output_mapping = {
                 0: "T-shirt",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat", 
                 5: "Sandal", 
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }

    #Extract examples from test set
    images, labels = [], []
    for idx in indices:
        image, label = testset[idx]
        images.append(image)
        labels.append(label)

    images = torch.stack(images)
    labels = torch.tensor(labels)

    #Flatten the images
    if model_type == 'ANN':
        images_flat = images.view(images.size(0), -1)
    else:
        images_flat = images

    #Get predictions
    with torch.no_grad():
        outputs = model(images_flat)
        _, predicted = torch.max(outputs, 1)

    #Plot the images with true and predicted labels
    num_examples = len(indices)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap='gray')
        true_label = output_mapping[labels[i].item()]
        predicted_label = output_mapping[predicted[i].item()]
        ax.set_title(f'True Label: {true_label}\nPredicted Label: {predicted_label}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    

def loss_plot(train_losses,test_losses,train_accuracy,test_accuracy,n_epochs=20):
    """Plots the losses and accuracies for train and test.
    
    Args:
    train_losses: Array of train losses of size n_epochs
    test_losses: Array of test losses of size n_epochs
    train_accuracy: Array of train accuracies of size n_epochs
    test_accuracy: Array of test accuracies of size n_epochs
    n_epochs: Number of epochs
    """
    n_epochs = list(range(1, n_epochs+1))
    
    # Plotting the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plotting loss
    ax1.plot(n_epochs, train_losses, label='Training Loss', color='red')
    ax1.plot(n_epochs, test_losses, label='Test Loss', color='orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss over Epochs')
    ax1.legend()
    ax1.grid(True)

    # Plotting accuracy
    ax2.plot(n_epochs, train_accuracy, label='Training Accuracy', color='blue')
    ax2.plot(n_epochs, test_accuracy, label='Test Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy over Epochs')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


    
    