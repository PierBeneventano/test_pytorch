'''Train neural nets with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

from models import *
import utils
import optim_util


parser = argparse.ArgumentParser(description='PyTorch Neural Networks')

# General arguments
parser.add_argument('--loss', choices=['l2', 'cross_entropy'], default='cross_entropy', help='what loss (criterion) to use')
parser.add_argument('--dataset', choices=['MNIST', 'cifar10', 'cifar100'], default='cifar10', help='what dataset to use')
parser.add_argument('--net', choices=['vgg', 'densenet', 'dla'], default='vgg', help='what model to train')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
parser.add_argument('--save_intermediate', choices=['yes', 'no'], default='yes', help='save the state at every epoch in which get better (it does not save in the previous 10)')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--save_jacobian_norm', choices=['yes', 'no'], default='no', 
                    help='save the average of the norm of the Jacobian on the first 20 points of the test set, and of 20 points randomly sampled from the training set')
# Optimizer related
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--optim_type', choices=['sgd', 'adam'], default='sgd')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
# Noise infection related
parser.add_argument('--gaussian_noise', default=0, type=float, help='the entity (the standard deviation sigma) of the gaussian noise.')
parser.add_argument('--input_gaussian_noise', default=0, type=float, help='the entity (the standard deviation sigma) of the input gaussian noise.')
parser.add_argument('--label_noise', default=0, type=float, help='probability of having label noise.')
parser.add_argument('--noise_sched', choices=['fixed', 'decay'], default='fixed',
					help='schedule of the label noise.')
parser.add_argument('--noise_decay', type=float, default=0.5,
					help='how much to multiply by, when we decay in label noise')

args = parser.parse_args()

# print the arguments
print("The arguments are: \n")
args = parser.parse_args()
for arg in vars(args):
	print(arg, " : ", getattr(args, arg))


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print('error: working with cpu')

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)



# initialize the vectors with the features to save

train_accuracy = np.zeros(args.epochs)
train_time = np.zeros(args.epochs)
train_loss = np.zeros(args.epochs)
test_accuracy = np.zeros(args.epochs)
test_loss = np.zeros(args.epochs)
ave_train_grad = np.zeros(args.epochs+1)
ave_test_grad = np.zeros(args.epochs+1)
max_train_grad = np.zeros(args.epochs+1)
max_test_grad = np.zeros(args.epochs+1)
# train_accuracy = []
# train_loss = []
# test_accuracy = []
# test_loss = []


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
    # torchvision.datasets.MNIST.resources = [
    # ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
    # ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
    # ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
    # ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
    # ]
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


# # plot 6 examples of data point
# examples = enumerate(testloader)
# batch_idx, (example_data, example_targets) = next(examples)
# example_data.shape

# for i in range(6):
#   print('the min entry in the example i is', torch.min(example_data[i][0]))
#   print('the max entry in the example i is', torch.max(example_data[i][0]))

# fig = plt.figure()
# for i in range(6):
#   plt.subplot(2,3,i+1)
#   plt.tight_layout()
#   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#   plt.title("Ground Truth: {}".format(example_targets[i]))
#   plt.xticks([])
#   plt.yticks([])
# fig


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
        net = VGG('VGG19', num_classes)

print('Number of model parameters: {}'.format(
		sum([p.data.nelement() for p in net.parameters()])))

# utils.init_params(net)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('/tigress/pb29/checkpoint/'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('/tigress/pb29/checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    args.net = checkpoint['architecture']

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
optimizer = optim_util.create_optimizer(net, optim_hparams)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)








# Training

iter_number_per_epoch = len(trainloader)
iter_number_per_epoch_test = len(testloader)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    starting_time_epoch = time.time()
    net.train()
    training_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # print(batch_idx, len(targets))

        # Label noise case
        if args.label_noise > 0:
            if args.noise_sched == 'fixed':
                label_noise = args.label_noise
            else:
                label_noise = optim_util.noise_decay(args.label_noise, epoch, args.noise_decay)
            
            targets = optim_util.apply_label_noise(targets, label_noise,
				num_classes=100 if args.dataset == 'cifar100' else 10)
        
        # Gaussian noise case
        if args.input_gaussian_noise > 0:
            if args.noise_sched == 'fixed':
                input_gaussian_noise = args.input_gaussian_noise
            else:
                input_gaussian_noise = optim_util.noise_decay(args.input_gaussian_noise, epoch, args.noise_decay)
            
            inputs = optim_util.apply_gaussian_noise(inputs, input_gaussian_noise)


        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Case of gaussian noise on gradient
        if args.gaussian_noise > 0:
            if args.noise_sched == 'fixed':
                gaussian_noise = args.gaussian_noise
            else:
                gaussian_noise = optim_util.noise_decay(args.gaussian_noise, epoch, args.noise_decay)
            orig_params = []
            for p in net.parameters():
	            orig_params.append(p.clone())
	            p.data = p.data.to(device) + gaussian_noise*torch.randn(p.size()).to(device)
                # p.data = (p.data + gaussian_noise*np.random.normal(loc=0, scale=torch.ones_like(p))).float()


        # Case of MNIST or cifar10
        if args.dataset == 'MNIST':
            outputs, aux = net(inputs)
        else:
            outputs = net(inputs)
        

        loss = criterion(outputs, targets)
        loss.backward()

        # Case of gaussian noise on gradient
        if args.gaussian_noise > 0:
            for p, orig_p in zip(net.parameters(), orig_params):
                p.data = orig_p.data

        optimizer.step()

        training_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


        # # Progress bar
        # utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (training_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save data of this iteration
    end_epoch = time.time()
    train_time[int(epoch)] = end_epoch-starting_time_epoch
    train_accuracy[int(epoch)] = 100.*correct/total
    train_loss[int(epoch)] = training_loss/iter_number_per_epoch
    
    # if epoch %10 == 0:
    #     # comput the gradient
    #     2+2




def test(epoch):
    global best_acc
    net.eval()
    iteration_test_loss = 0
    correct = 0
    total = 0
    last_saved=0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if args.dataset == 'MNIST':
                outputs, aux = net(inputs)
            else:
                outputs = net(inputs)
            loss = criterion(outputs, targets)

            iteration_test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # utils.progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (iteration_test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    test_accuracy[int(epoch)] = 100.*correct/total
    test_loss[int(epoch)] = iteration_test_loss/iter_number_per_epoch_test

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        if args.save_intermediate == 'yes':
            if (int(epoch)-last_saved) >= 10 :
                last_saved = int(epoch)
                print('Saving...')
                state = {
                    'acc': acc,
                    'epoch': epoch+1,
                    'architecture':args.net,
                    'dataset': args.dataset,
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                if not os.path.isdir('/tigress/pb29/checkpoint/training/dataset_{}-model_{}'.format(args.dataset, args.net)):
                    os.mkdir('/tigress/pb29/checkpoint/training/dataset_{}-model_{}'.format(args.dataset, args.net))
                torch.save(state, '/tigress/pb29/checkpoint/ckpt.pt')
                torch.save(state, '/tigress/pb29/checkpoint/training/dataset_{}-model_{}/epoch_{}-label_noise_prob_{}-input_gaussian_noise_SD_{}-gaussian_noise_SD_{}-noise_decay_{}-batch_size_{}.pt'
                        .format(args.dataset, args.net, epoch+1, args.label_noise, args.input_gaussian_noise, args.gaussian_noise, args.noise_sched, args.batchsize))



np.random.seed(0)
torch.manual_seed(0)


for epoch in range(start_epoch, args.epochs):

    if args.save_jacobian_norm == 'yes':
        examples = enumerate(trainloader)
        examples_test = enumerate(testloader)
        batch_idx, (example_data, example_targets) = next(examples)
        example_data.shape
        batch_idx_1, (example_test_data, example_test_targets) = next(examples_test)
        example_test_data.shape
        for i in range(1,21):
            jac_norm_train = torch.norm(torch.autograd.functional.jacobian(net, example_data[i:i+1,:,:,:], create_graph = True))
            jac_norm_test = torch.norm(torch.autograd.functional.jacobian(net, example_test_data[i:i+1,:,:,:], create_graph = True))
            ave_train_grad[epoch] = ave_train_grad[epoch] + jac_norm_train
            ave_test_grad[epoch] = ave_test_grad[epoch] + jac_norm_test
            max_train_grad[epoch] = max(max_train_grad[epoch], jac_norm_train)
            max_test_grad[epoch] = max(max_test_grad[epoch], jac_norm_test)
        ave_train_grad[epoch] = ave_train_grad[epoch]/20
        ave_test_grad[epoch] = ave_test_grad[epoch]/20

    train(epoch)
    test(epoch)
    

    print('Training accuracy:',
    train_accuracy[int(epoch)], ', training loss: ', train_loss[int(epoch)], ', time of the epoch: ', train_time[int(epoch)],
    's, test accuracy: ', test_accuracy[int(epoch)], ', test loss: ', test_loss[int(epoch)])
    scheduler.step()

# Final saving

if args.save_jacobian_norm == 'yes':
    examples = enumerate(trainloader)
    examples_test = enumerate(testloader)
    batch_idx, (example_data, example_targets) = next(examples)
    example_data.shape
    batch_idx_1, (example_test_data, example_test_targets) = next(examples_test)
    example_test_data.shape
    for i in range(1,21):
        jac_norm_train = torch.norm(torch.autograd.functional.jacobian(net, example_data[i:i+1,:,:,:], create_graph = True))
        jac_norm_test = torch.norm(torch.autograd.functional.jacobian(net, example_test_data[i:i+1,:,:,:], create_graph = True))
        ave_train_grad[args.epochs+1] = ave_train_grad[args.epochs+1] + jac_norm_train
        ave_test_grad[args.epochs+1] = ave_test_grad[args.epochs+1] + jac_norm_test
        max_train_grad[args.epochs+1] = max(max_train_grad[args.epochs+1], jac_norm_train)
        max_test_grad[args.epochs+1] = max(max_test_grad[args.epochs+1], jac_norm_test)
    ave_train_grad[args.epochs+1] = ave_train_grad[args.epochs+1]/20
    ave_test_grad[args.epochs+1] = ave_test_grad[args.epochs+1]/20


print('Final saving...')
state = {
    'best_acc': best_acc,
    'epoch': epoch+1,
    'architecture':args.net,
    'dataset': args.dataset,
    'net': net.state_dict(),
    'optimizer': optimizer.state_dict(),
    'test_acc_array': test_accuracy,
    'test_loss_array': test_loss,
    'train_acc_array': train_accuracy,
    'train_loss_array': train_loss,
    'train_time': train_time,
    'average_train_grad': ave_train_grad,
    'average_test_grad': ave_test_grad,
    'max_train_grad': ave_train_grad,
    'max_test_grad': ave_test_grad,
}
if not os.path.isdir('/tigress/pb29/checkpoint/final/dataset_{}'.format(args.dataset)):
    os.mkdir('/tigress/pb29/checkpoint/final/dataset_{}'.format(args.dataset))
torch.save(state, '/tigress/pb29/checkpoint/final/dataset_{}/FINAL_-model_{}-epoch_{}-label_noise_prob_{}-input_gaussian_noise_{}-gaussian_noise_SD_{}-noise_decay_{}-batch_size_{}.pt'
            .format(args.dataset, args.net, args.epochs, args.label_noise, args.input_gaussian_noise, args.gaussian_noise, args.noise_sched, args.batchsize))
