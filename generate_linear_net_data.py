# Dataset generation from linear nets
import torch
import torch.nn as nn
import torch.nn.functional as F

import os

class LinearNetData(torch.utils.data.Dataset):
    def __init__(self, list_IDs, labels, inputs):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.inputs = inputs
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = self.inputs[ID]
        y = self.labels[ID]

        return X, y


class lin_generator(nn.Module):
    def __init__(self):
        super(lin_generator, self).__init__()
        self.l1 = nn.Linear(784, 256)
        self.l2 = nn.Linear(256, 64)
        self.l3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        output = F.log_softmax(x, dim=0)
        return output
    

if __name__ == '__main__':

    partition = {}
    partition['training'] = [*range(0, 50000)]
    partition['test'] = [*range(50000, 60000)]

    net = lin_generator()

    x = {}
    y = {}
    z = {}

    for ID in range(0,50000):
        x[ID] = torch.randn(784)
        z[ID] = net(x[ID])
        y[ID] = torch.argmax(z[ID])
    
    train_data = LinearNetData(list_IDs = partition['training'], labels = y, inputs = x)

    x = {}
    y = {}
    z = {}

    for ID in range(0,10000):
        x[ID] = torch.randn(784)
        z[ID] = net(x[ID])
        y[ID] = torch.argmax(z[ID])
    
    test_data = LinearNetData(list_IDs = partition['test'], labels = y, inputs = x)

    state = {
        'generator linear net': net,
    }
    print('a')
    torch.save(state, 'data/synthetic_linear_net/generator.pt')
    print('b')
    torch.save(train_data, 'data/synthetic_linear_net/training.pt')
    print('c')
    torch.save(test_data, 'data/synthetic_linear_net/test.pt')
    print('d')

