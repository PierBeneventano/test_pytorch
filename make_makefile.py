import os
import numpy as np
cwd = os.getcwd()   # get the current working directory

def create_makefile(choice_dict):
    f = open(f"{cwd}/makefile","w")
    f.write(".PHONY = help setup test run clean environment\n\n")
    f.write(".DEFAULT_GOAL = setup\n")
    f.write("setup:\n")

    # plain sgd
    #f.write(f"\t@python main.py --dataset 'MNIST'\n")
    for net_choice in choice_dict['net']:
        f.write(f"\t@python main.py --net {net_choice}\n")

    # full_batch
    #f.write(f"\t@python main.py --dataset 'MNIST' --batchsize 1024\n")
    for net_choice in choice_dict['net']:
        f.write(f"\t@python main.py --net {net_choice} --batchsize 1024\n")

    # Gradient noise
    for i in [128, 1024]:
        for lg_choice in choice_dict['gaussian_noise_sigma']:
            for lg_sched_choice in choice_dict['noise_sched']:
                #f.write(f"\t@python main.py --dataset 'MNIST' --gaussian_noise {lg_choice} --noise_sched {lg_sched_choice} --batchsize {i} \n")
                for net_choice in choice_dict['net']:
                    f.write(f"\t@python main.py --net {net_choice} --gaussian_noise {lg_choice} --noise_sched {lg_sched_choice} --batchsize {i} \n")

    # Imput Gaussian noise
    for i in [128, 1024]:
        for lg_choice in choice_dict['gaussian_noise_sigma']:
            for lg_sched_choice in choice_dict['noise_sched']:
                f.write(f"\t@python main.py --dataset 'MNIST' --input_gaussian_noise {lg_choice} --noise_sched {lg_sched_choice} --batchsize {i} \n")
                for net_choice in choice_dict['net']:
                    f.write(f"\t@python main.py --net {net_choice} --input_gaussian_noise {lg_choice} --noise_sched {lg_sched_choice} --batchsize {i} \n")

    # SGD + label noise
    for i in [128, 1024]:
        for ln_choice in choice_dict['label_noise_prob']:
            for ln_sched_choice in choice_dict['noise_sched']:
                f.write(f"\t@python main.py --dataset 'MNIST' --label_noise {ln_choice} --noise_sched {ln_sched_choice} --batchsize {i} \n")
                for net_choice in choice_dict['net']:
                    f.write(f"\t@python main.py --net {net_choice} --label_noise {ln_choice} --noise_sched {ln_sched_choice} --batchsize {i} \n")
    

if __name__ == "__main__":
    # creating choice dictionary
    choice_dict = {}

    net_choices = np.array(['vgg', 'densenet', 'dla'])
    label_noise_choices = np.array([0.5, 0.2, 0.1])
    g_noise_choices = np.array([0.2, 0.1, 0.02])
    noise_sched_choices = np.array(['decay', 'fixed'])

    choice_dict['net'] = net_choices
    choice_dict['label_noise_prob'] = label_noise_choices
    choice_dict['noise_sched'] = noise_sched_choices
    choice_dict['gaussian_noise_sigma'] = g_noise_choices

    create_makefile(choice_dict)
    
