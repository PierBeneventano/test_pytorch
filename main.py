'''Train neural nets with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import numpy as np

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import os
import argparse

from models import *
import utils
import optim_util


parser = argparse.ArgumentParser(description='PyTorch Neural Networks')

# General arguments
parser.add_argument('--loss', choices=['l2', 'cross_entropy'], default='cross_entropy', help='what loss (criterion) to use')
parser.add_argument('--dataset', choices=['MNIST', 'cifar10', 'cifar100'], default='cifar10', help='what dataset to use')
parser.add_argument('--net', choices=['vgg', 'densenet', 'dla'], default='vgg', help='what model to train')
parser.add_argument('--number_epochs', default=200, type=int, help='number of epoxhs')
parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
parser.add_argument('--save_intermediate', choices=['yes', 'no'], default='yes', help='save the state at every epoch in which get better or not')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# Optimizer related
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--optim_type', choices=['sgd', 'adam'], default='sgd')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
					help='weight decay (default: 5e-4)')
# Label noise related
parser.add_argument('--label_noise', default=0, type=float, help='probability of having label noise.')
parser.add_argument('--ln_sched', choices=['fixed', 'decay'], default='fixed',
					help='schedule of the label noise.')
parser.add_argument('--ln_decay', type=float, default=0.5,
					help='how much to multiply by when we decay')

args = parser.parse_args()

# print the arguments
print("The arguments are: \n")
args = parser.parse_args()
for arg in vars(args):
	print(arg, " : ", getattr(args, arg))


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data I want all the data pipeline in another file
print('==> Preparing data..')
if args.dataset == 'cifar10':
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    crop=32
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
    input_dim = [1,3,32,32]
    num_classes = 10

elif args.dataset == 'cifar100':
    stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
    crop=32
    input_dim = [1,3,32,32]
    num_classes = 100
    
elif args.dataset == 'MNIST':
    stats = ((0.1307,), (0.3081,))
    crop=28
    input_dim = [1,1,28,28]
    num_classes = 10

transform_train = transforms.Compose([
    transforms.RandomCrop(crop, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*stats),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats),
])

if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, args.batchsize, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, args.batchsize, shuffle=False, num_workers=2)

elif args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, args.batchsize, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, args.batchsize, shuffle=False, num_workers=4)

elif args.dataset == 'MNIST':
    torchvision.datasets.MNIST.resources = [
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
    ]
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, 
        transform=torchvision.transforms.Compose([
            transforms.RandomRotation(5, fill=(0,)),
            transforms.RandomCrop(crop, padding = 2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*stats)]))
        
    trainloader = torch.utils.data.DataLoader(
        trainset, args.batchsize, shuffle=True)

    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, 
        transform=torchvision.transforms.Compose([
            transforms.RandomRotation(5, fill=(0,)),
            transforms.RandomCrop(crop, padding = 2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*stats)]))
    testloader = torch.utils.data.DataLoader(
        testset, args.batchsize, shuffle=False)


# plot 6 examples of data point
examples = enumerate(testloader)
batch_idx, (example_data, example_targets) = next(examples)
example_data.shape

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
fig


# Model
print('==> Building model..')
if args.dataset == 'MNIST':
    INPUT_DIM =28*28
    net = MLP(INPUT_DIM, num_classes)
    args.net = 'MLP'
else:
    if args.net == 'densenet':
        net = DenseNet121()
    elif args.net == 'dla':
        net = DLA()
    else:
        net = VGG('VGG19')

print('Number of model parameters: {}'.format(
		sum([p.data.nelement() for p in net.parameters()])))

utils.init_params(net)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

if args.loss == 'cross_entropy':
	criterion = nn.CrossEntropyLoss().cuda()
else:
	criterion = nn.MSELoss().cuda()
    # change something to the model in this case







# The optimization
optim_hparams = {
	'base_lr' : args.lr, 
	'momentum' : args.momentum,
	'weight_decay' : args.weight_decay,
	'optim_type' : args.optim_type
}
optimizer = optim_util.create_optimizer(
	net,	optim_hparams)

# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)








# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # print('a', batch_idx, len(targets))
        
        #Label noise case
        if args.label_noise > 0:
            if args.ln_sched == 'fixed':
                label_noise = args.label_noise
            else:
                label_noise = optim_util.ln_decay(args.label_noise, epoch, args.ln_decay)
            
            targets = optim_util.apply_label_noise(targets, label_noise,
				num_classes=10 if args.dataset == 'cifar10' else 100)

        inputs, targets = inputs.to(device), targets.to(device)
        # print('b')
        optimizer.zero_grad()
        # print('c')
        if args.dataset == 'MNIST':
            outputs, aux = net(inputs)
        else:
            outputs = net(inputs)
        # print('d')
        loss = criterion(outputs, targets)
        # print('e')
        loss.backward()
        # print('f')
        optimizer.step()
        # print('g')

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))



def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if args.dataset == 'MNIST':
                outputs, aux = net(inputs)
            else:
                outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            utils.progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving...')
        state = {
            'acc': acc,
            'epoch': epoch+1,
            'architecture':args.net,
            'dataset': args.dataset,
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pt')
        torch.save(state, './checkpoint/dataset:{}-model:{}-epoch:{}.pt'.format(args.dataset, args.net, epoch+1))
        best_acc = acc


for epoch in range(start_epoch, args.number_epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
