import torch
import numpy
import matplotlib.pyplot as plt

print('==> Resuming from checkpoint..')
checkpoint = torch.load('/tigress/pb29/checkpoint/final/FINAL_dataset_MNIST-model_MLP-epoch_200-label_noise_prob_0-input_gaussian_noise_0-gaussian_noise_SD_0-noise_decay_fixed-batch_size_128.pt',
                         map_location ='cpu')
best_acc = checkpoint['best_acc']
epoch = checkpoint['epoch']
net_arch = checkpoint['architecture']
dataset = checkpoint['dataset']
net = checkpoint['net']
test_accuracy = checkpoint['test_acc_array']
train_accuracy = checkpoint['train_acc_array']
test_loss = checkpoint['test_loss_array']
train_loss = checkpoint['train_loss_array']
train_time = checkpoint['train_time']

print(dataset)
print(best_acc)
# plt.plot(train_time)
# plt.plot(test_loss)
# plt.plot(test_accuracy)
# plt.plot(train_loss)
# plt.plot(train_accuracy)

print('train time', train_time)
print('train loss', train_loss)
print('train accuracy', train_accuracy)
print('test loss', test_loss)
print('test accuracy', test_accuracy)

# plt.plot(test_loss)
# plt.plot(test_accuracy)
# plt.plot(train_loss)
# plt.plot(train_accuracy)
