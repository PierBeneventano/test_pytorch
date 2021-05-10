import torch
import torch.optim as optim
import numpy as np

def create_optimizer(model, hparams):
	if hparams['optim_type'] == 'sgd':
		if hparams['momentum'] > 0:
			return optim.SGD(
				[{'params': model.parameters(), 'lr': hparams['base_lr'], 'initial_lr': hparams['base_lr']}], 
				momentum=hparams['momentum'],
				weight_decay=hparams['weight_decay'],
				nesterov=True
				)
		else:
			return optim.SGD(
				[{'params': model.parameters(), 'lr': hparams['base_lr'], 'initial_lr': hparams['base_lr']}], 
				momentum=hparams['momentum'],
				weight_decay=hparams['weight_decay'],
				nesterov=False
				)
	elif hparams['optim_type'] == 'adam':
		return optim.AdamW(
			[{'params': model.parameters(), 'lr': hparams['base_lr'], 'initial_lr' : hparams['base_lr']}],	
			weight_decay=hparams['weight_decay'])



# Label noise utility

def apply_label_noise(labels, noise_prob, num_classes=10):
	new_labels = labels
	for i in new_labels:
		if np.random.uniform(0,1) < noise_prob:
			i = np.random.randint(low=0, high=num_classes)
	return new_labels

def apply_gaussian_noise(inputs, sigma):
	new_inputs = inputs
	for i in new_inputs:
			i = i + torch.randn(1)*sigma
	return new_inputs

def noise_decay(noise, epoch, noise_decay=0.5):
	noise_new = noise*(noise_decay**int(epoch >= 150))
	noise_new *= (noise_decay**int(epoch >= 250))
	return noise_new