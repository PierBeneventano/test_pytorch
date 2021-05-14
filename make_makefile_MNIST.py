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
    
            for net_choice in choice_dict['net']:
                f.write(f"\t@python main.py main_MNIST.py --net {net_choice} --lr {lr_choices} --batchsize {batch_size}\n")


    # Gradient noise
    for lg_choice in choice_dict['gaussian_noise_sigma']:
        for lg_sched_choice in choice_dict['noise_sched']:
            f.write(f"\t@python main.py --dataset 'MNIST' --gaussian_noise {lg_choice} --noise_sched {lg_sched_choice} --batchsize 128 \n")
            f.write(f"\t@python main.py  --gaussian_noise {lg_choice} --noise_sched {lg_sched_choice} --batchsize 128 \n")


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
    
