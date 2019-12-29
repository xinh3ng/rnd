"""
Tutorial: https://towardsdatascience.com/an-easy-introduction-to-pytorch-for-neural-networks-3ea08516bff2

# Usage
export PYTHONPATH=$(pwd)
"""
import json
import os
import pandas as pd
from pypchutils.generic import create_logger
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms

logger = create_logger(__name__)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        """Set up the model by stacking all the layers together
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = self.global_pool(x)

        x = x.view(-1, 64)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        x = F.log_softmax(x)
        return x


#############

num_epochs = 10
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root="/tmp/data", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root="/tmp/data", train=False, transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Set up the model
model = Net()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

# Train the model
total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward step
        outputs = model(images)
        loss = loss_function(outputs, labels)

        # Backward step and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            logger.info(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{total_steps}], Loss: {loss.item()}")

torch.save(model.state_dict(), "/tmp/model.ckpt")
