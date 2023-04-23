### YOUR CODE HERE
# %%
import numpy as np
import torch
import torch.nn.functional as F
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

#Initialize parameters in matrix form - with AD
W1 = torch.randn(n0,n1)/np.sqrt(n0)
W1.requires_grad_()
W2 = torch.randn(n1,n2)/np.sqrt(n1)
W2.requires_grad_()
b1 = torch.zeros(n1, requires_grad=True)
b2 = torch.zeros(n2, requires_grad=True)

#Create optimizer
optimizer = torch.optim.SGD([W1,W2,b1,b2], lr=0.3)

# %%

#Define number of epochs
num_epochs = 1

#Perform training for multiple epochs
for epoch in range(num_epochs):

    #Perform training 
    for images, labels in tqdm(train_loader):
        # Zero out the gradients
        optimizer.zero_grad()

        # Flatten input x
        x = images.view(-1, n0)
        # Hidden layer
        h = torch.relu(torch.matmul(x,W1)+b1)
        # Output layer
        y = torch.matmul(h,W2)+b2
        # Cross-entropy
        cross_entropy = F.cross_entropy(y, labels)
        # Backward
        cross_entropy.backward()
        optimizer.step()

# %%

## Testing
correct = 0
total = len(mnist_test)

with torch.no_grad():    
    for images, labels in tqdm(test_loader):

        # Flatten input x
        x = images.view(-1, n0)
        # Hidden layer
        h = torch.relu(torch.matmul(x,W1)+b1)
        # Output layer
        y = torch.matmul(h,W2)+b2
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())

# Make sure to print out your accuracy on the test set at the end.
print('Test accuracy: {}'.format(correct/total))


      

# %%
