'''Linear networks in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNet(nn.Module):
# I want to give as input the input dimension and the output dimension
    def __init__(self, depth=5, input_dim = [1,3,32,32], num_classes=10):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, 512)
        for i in range(depth-2):
            self.lin = nn.Linear(512, 512)
        self.classifier = nn.Linear(512, num_classes)


def Linearnet_cifar_5():
    return LinearNet()

def Linearnet_cifar_10():
    return LinearNet(depth=10)

def Linearnet_cifar_20():
    return LinearNet(depth=20)

def Linearnet_MNIST_5():
    return LinearNet(input_dim = [1,28,28])

def test():
    net = linearnet_cifar_10()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

# test()
