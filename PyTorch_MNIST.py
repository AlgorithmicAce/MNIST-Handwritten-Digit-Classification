#Importing Required Libraries
import torch
from torch import nn, optim, relu
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as tforms

#device = torch.device('cuda:0')
#Uncomment the code above if you have a CUDA supported GPU and wants to use it to speed up the training process

#Download the data from the web
train_dataset = dsets.MNIST(root = './resources/data', train = True, download = True, transform = tforms.ToTensor())
test_dataset = dsets.MNIST(root = './resources/data', train = False, download = True, transform = tforms.ToTensor())

#Loads the data
train_loader = DataLoader(dataset = train_dataset, batch_size = 3000)
test_loader = DataLoader(dataset = test_dataset, batch_size = 500)

#Creating a class
class Net(nn.Module):
    def __init__(self, din, h1,dout):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(din, h1)
        self.linear2 = nn.Linear(h1, dout)

    def forward(self, x):
        x = self.linear1(x)
        x = relu(x)
        x = self.linear2(x)
        return x

#Creating a neural network model with 784 input neurons, 28 neurons in the hidden layer and 10 output neurons
model = Net(784, 28, 10)

#model.to(device)
#Uncomment the code above if you have a CUDA supported GPU and wants to use it to speed up the training process

#Using Adam optimizer with a learning rate of 0.1 and creating a Cross Entropy Loss Function
optimizer = optim.Adam(model.parameters(), lr = 0.1)
criterion = nn.CrossEntropyLoss()

#Trains the model
n_epoch = 35
loss_list = []
accuracy_list = []
test_number = len(test_dataset)

for epoch in range(n_epoch):
    for x, y in train_loader:
        #x, y = x.to(device), y.to(device)
        x = x.view(-1, 28 * 28)
        yhat = model(x)
        loss = criterion(yhat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    correct = 0
    for x_test, y_test in test_loader:
        #x_test, y_test = x_test.to(device), y_test.to(device)
        x_test = x_test.view(-1, 28 * 28)
        z_test = model(x_test)
        zval, zidx = z_test.max(1)
        correct += (zidx == y_test).sum().item()
    accuracy = correct / test_number
    loss_list.append(loss.item())
    accuracy_list.append(accuracy)

plt.plot(accuracy_list, label = 'Accuracy')
plt.title('Graph of Accuracy against Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Accuracy without CNN.png', format = 'png')
plt.show()

plt.plot(loss_list, label = 'Loss')
plt.title('Graph of Loss against Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Loss without CNN.png', format = 'png')
plt.show()