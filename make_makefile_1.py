import os
import numpy as np
cwd = os.getcwd()   # get the current working directory

def create_makefile(choice_dict):
    f = open(f"{cwd}/makefile",'w')
    f.write(".PHONY = help setup test run clean environment\n\n")
    f.write(".DEFAULT_GOAL = setup\n")
    f.write("setup:\n")

    # plain sgd
    f.write(f"\t@python main_MNIST.py --batchsize 50000\n")
    # f.write(f"\t@python main_MNIST.py --net linear --batchsize 50000\n")

    # # Gradient noise
    # for lg_choice in choice_dict['gaussian_noise_sigma']:
    #     for lg_sched_choice in choice_dict['noise_sched']:
    #         f.write(f"\t@python3 main.py  --gaussian_noise {lg_choice} --noise_sched {lg_sched_choice} --batchsize 265 \n")
    #         f.write(f"\t@python3 main.py  --gaussian_noise {lg_choice} --noise_sched {lg_sched_choice} --batchsize 64 \n")


if __name__ == "__main__":
    # creating choice dictionary
    choice_dict = {}

    net_choices = np.array(['vgg', 'densenet', 'dla'])
    label_noise_choices = np.array([0.5, 0.2, 0.1])
    g_noise_choices = np.array([0.2, 0.1, 0.02])
    noise_sched_choices = np.array(['decay', 'fixed'])
    lr_choices = np.array([0.1, 0.01, 0.001])

    choice_dict['net'] = net_choices
    choice_dict['label_noise_prob'] = label_noise_choices
    choice_dict['noise_sched'] = noise_sched_choices
    choice_dict['gaussian_noise_sigma'] = g_noise_choices
    choice_dict['learning_rate'] = lr_choices

    create_makefile(choice_dict)
    
