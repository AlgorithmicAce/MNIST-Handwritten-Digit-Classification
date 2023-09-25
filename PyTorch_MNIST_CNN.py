import torch
from torch import nn, optim, relu
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as tforms
#device = torch.device('cuda:0')

composetransform = tforms.Compose([
    tforms.Resize((20, 20)),
    tforms.ToTensor()
])

train_dataset =  dsets.MNIST(root = './resource/data', train = True, download = True, transform = composetransform)
test_dataset = dsets.MNIST(root = './resources/data', train = False, download = True, transform = composetransform)
train_loader = DataLoader(dataset = train_dataset, batch_size = 3000)
test_loader = DataLoader(dataset = test_dataset, batch_size = 500)

class CNN(nn.Module):
    def __init__(self, out1, out2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = out1, kernel_size = 5, stride = 1, padding = 0)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 1, padding = 0)

        self.conv2 = nn.Conv2d(in_channels = out1, out_channels = out2, kernel_size = 5, stride = 1, padding = 0)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 1, padding = 0)
        self.fc1 = nn.Linear(out2 * 10 * 10, 10)

    def forward(self,x):
        x = relu(self.conv1(x))
        x = self.maxpool1(x)
        x = relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

model = CNN(4, 16)
#model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

#Trains the model
n_epoch = 5
loss_list = []
accuracy_list = []
test_number = len(test_dataset)

for epoch in range(n_epoch):
    for x, y in train_loader:
        #x, y = x.to(device), y.to(device)
        yhat = model(x)
        loss = criterion(yhat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    correct = 0
    for x_test, y_test in test_loader:
        #x_test, y_test = x_test.to(device), y_test.to(device)
        z_test = model(x_test)
        zval, zidx = z_test.max(1)
        correct += (zidx == y_test).sum().item()
    accuracy = correct / test_number
    loss_list.append(loss.item())
    accuracy_list.append(accuracy)

plt.plot(accuracy_list, label = 'Accuracy')
plt.title('Graph of Accuracy against Epoch with CNN')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Accuracy with CNN.png', format = 'png')
plt.show()

print("The final accuracy with CNN is",accuracy_list[-1],"\n")

plt.plot(loss_list, label = 'Loss')
plt.title('Graph of Loss against Epoch with CNN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Loss with CNN.png', format = 'png')
plt.show()

print("The final loss with CNN is",loss_list[-1],"\n")