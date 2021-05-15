import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear_mnist(nn.Module):
    def __init__(self):
        super(Linear_mnist, self).__init__()
        self.l1 = nn.Linear(784, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 256)
        self.l4 = nn.Linear(256, 256)
        self.l5 = nn.Linear(256, 128)
        self.l6 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.l1(x)
        # x = F.relu(x)
        x = self.l2(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        # x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        output = F.log_softmax(x, dim=1)
        return output