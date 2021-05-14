import os
import numpy as np
cwd = os.getcwd()   # get the current working directory

def create_makefile(choice_dict):
    f = open(f"{cwd}/makefile",'w')
    f.write(".PHONY = help setup test run clean environment\n\n")
    f.write(".DEFAULT_GOAL = setup\n")
    f.write("setup:\n")

    for batch_size in [64, 128, 1024, 4096]:
        for lr_choices in choice_dict['learning_rate']:
            f.write(f"\t@python main_MNIST.py --lr {lr_choices} --batchsize {batch_size} --epochs 200\n")
            for ln_choice in choice_dict['label_noise_prob']:
                f.write(f"\t@python main_MNIST.py --label_noise {ln_choice} --lr {lr_choices} --batchsize {batch_size} --epochs 200\n")
            for lg_sched_choice in choice_dict['noise_sched']:
                f.write(f"\t@python main_MNIST.py --input_gaussian_noise {ln_choice} --lr {lr_choices} --batchsize {batch_size} --epochs 200\n")


if __name__ == "__main__":
    # creating choice dictionary
    choice_dict = {}

    net_choices = np.array(['MLP', 'linear', 'conv'])
    label_noise_choices = np.array([0.5, 0.2, 0.1])
    g_noise_choices = np.array([0.2, 0.1, 0.02])
    noise_sched_choices = np.array(['decay', 'fixed'])
    lr_choices = np.array([0.1, 0.01, 1])

    choice_dict['net'] = net_choices
    choice_dict['label_noise_prob'] = label_noise_choices
    choice_dict['noise_sched'] = noise_sched_choices
    choice_dict['gaussian_noise_sigma'] = g_noise_choices
    choice_dict['learning_rate'] = lr_choices

    create_makefile(choice_dict)
    
