import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_mnist(nn.Module):
    def __init__(self):
        super(CNN_mnist, self).__init__()
        self.l1 = nn.Linear(1, 32, 3, 1)
        self.l2 = nn.Linear(32, 64, 3, 1)
        self.l5 = nn.Linear(9216, 128)
        self.l6 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output