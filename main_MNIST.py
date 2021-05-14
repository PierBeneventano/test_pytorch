# October 25, 2020
# https://github.com/pytorch/examples/blob/master/mnist/main.py

from __future__ import print_function
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import optim_util
from models import *


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batchsize', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
parser.add_argument('--test_batchsize', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')


# Noise injection related
parser.add_argument('--gaussian_noise', default=0, type=float, help='the entity (the standard deviation sigma) of the gaussian noise.')
parser.add_argument('--input_gaussian_noise', default=0, type=float, help='the entity (the standard deviation sigma) of the input gaussian noise.')
parser.add_argument('--label_noise', default=0, type=float, help='probability of having label noise.')
parser.add_argument('--noise_sched', choices=['fixed', 'decay'], default='fixed',
                help='schedule of the label noise.')
parser.add_argument('--noise_decay', type=float, default=0.5,
                help='how much to multiply by, when we decay in label noise')

args = parser.parse_args()


best_acc = 0
train_accuracy = np.zeros(args.epochs)
train_time = np.zeros(args.epochs)
train_loss = np.zeros(args.epochs)
test_accuracy = np.zeros(args.epochs)
test_loss = np.zeros(args.epochs)


def train(args, model, device, train_loader, optimizer, epoch):
    
    training_loss = 0
    correct = 0
    total = 0
    starting_time_epoch = time.time()
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # label noise
        if args.label_noise > 0:
            if args.noise_sched == 'fixed':
                label_noise = args.label_noise
            else:
                label_noise = optim_util.noise_decay(args.label_noise, epoch, args.noise_decay)
            
            target = optim_util.apply_label_noise(target, label_noise, 10)
        
        # Gaussian noise case
        if args.input_gaussian_noise > 0:
            if args.noise_sched == 'fixed':
                input_gaussian_noise = args.input_gaussian_noise
            else:
                input_gaussian_noise = optim_util.noise_decay(args.input_gaussian_noise, epoch, args.noise_decay)
            
            data = optim_util.apply_gaussian_noise(data, input_gaussian_noise)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        training_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
        #     if args.dry_run:
        #         break

    end_epoch = time.time()
    train_time[int(epoch)] = end_epoch-starting_time_epoch
    train_accuracy[int(epoch)] = 100.*correct/total
    train_loss[int(epoch)] = training_loss/total



def test(model, device, test_loader, epoch):
    model.eval()
    iteration_test_loss = 0
    correct = 0
    last_saved = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            iteration_test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    iteration_test_loss /= len(test_loader.dataset)

    print('\nEpoch {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch,
        iteration_test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_accuracy[int(epoch)] = 100.*correct/len(test_loader.dataset)
    test_loss[int(epoch)] = iteration_test_loss/len(test_loader.dataset)



    acc = 100.*correct/len(test_loader.dataset)
    if (int(epoch)-last_saved) >= 10 :
        last_saved = int(epoch)
        if acc > best_acc:
            print('Saving...')
            state = {
                'acc': acc,
                'epoch': epoch+1,
                'architecture': 'CNN_mnist',
                'dataset': 'MNIST',
                'net': model.state_dict(),
            }
            if not os.path.isdir('/tigress/pb29/checkpoint/training/dataset_MNIST-model_CNN'):
                os.mkdir('/tigress/pb29/checkpoint/training/dataset_MNIST-model_CNN')
            torch.save(state, '/tigress/pb29/checkpoint/ckpt.pt')
            torch.save(state, '/tigress/pb29/checkpoint/training/dataset_MNIST-model_CNN/epoch_{}-label_noise_prob_{}-input_gaussian_noise_SD_{}-gaussian_noise_SD_{}-noise_decay_{}-batch_size_{}-lr_{}.pt'
                        .format(args.epochs, args.label_noise, args.input_gaussian_noise, args.gaussian_noise, args.noise_sched, args.batchsize, args.lr))
            best_acc = acc


def main():
    # Training settings
    print("The arguments are: \n")
    args = parser.parse_args()
    for arg in vars(args):
	    print(arg, " : ", getattr(args, arg))

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batchsize}
    test_kwargs = {'batch_size': args.test_batchsize}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)


    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('data', train=True, download=False,
                       transform=transform)
    dataset2 = datasets.MNIST('data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = CNN_mnist().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)



    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch)
        scheduler.step()




    print('Final saving...')
    state = {
        'best_acc': 1,
        'epoch': epoch+1,
        'architecture':'CNN_mnist',
        'dataset': args.dataset,
        'net': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'test_acc_array': test_accuracy,
        'test_loss_array': test_loss,
        'train_acc_array': train_accuracy,
        'train_loss_array': train_loss,
        'train_time': train_time,
    }   
    torch.save(state, '/tigress/pb29/checkpoint/final/FINAL_dataset_MNIST-model_CNN-epoch_{}-label_noise_prob_{}-input_gaussian_noise_{}-gaussian_noise_SD_{}-noise_decay_{}-batch_size_{}-lr_{}.pt'
            .format(args.epochs, args.label_noise, args.input_gaussian_noise, args.gaussian_noise, args.noise_sched, args.batchsize, args.lr))


if __name__ == '__main__':
    main()