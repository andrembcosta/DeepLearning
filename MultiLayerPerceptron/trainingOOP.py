### OBJECT-ORIENTED VERSION
# %%
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm

# Load the data
mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
# Make train and test sets and mini batches
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

#Set sizes of layers
n0 = 28*28
n1 = 500
n2 = 10

class Net(nn.Module):
    def __init__(self, n0, n1, n2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n0, n1)
        self.fc2 = nn.Linear(n1, n2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#Initialize model
model = Net(n0, n1, n2)

#Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.3)

# %%
#Define number of epochs
num_epochs = 1

#Perform training for multiple epochs
for epoch in range(num_epochs):
    #Perform training 
    for images, labels in tqdm(train_loader):
        # Zero out the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images.view(-1, n0))

        # Compute loss and backpropagation
        loss = criterion(outputs, labels)
        loss.backward()

        # Update weights
        optimizer.step()


# %%
## Testing - OBJECT-ORIENTED VERSION
correct = 0
total = len(mnist_test)

with torch.no_grad():    
    for images, labels in tqdm(test_loader):

        # Forward pass
        y = model(images.view(-1, n0))
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())

# Make sure to print out your accuracy on the test set at the end.
print('Test accuracy: {}'.format(correct/total))        
# %%
