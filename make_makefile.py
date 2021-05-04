import os
import numpy as np
cwd = os.getcwd()   # get the current working directory

def create_makefile(choice_dict):
    f = open(f"{cwd}/makefile","w")
    f.write(".PHONY = help setup test run clean environment\n\n")
    f.write(".DEFAULT_GOAL = setup\n")
    f.write("setup:\n")
    for net_choice in choice_dict['net']:
        for dataset_choice in choice_dict['dataset']:
            f.write(f"\t@python main.py --net {net_choice} --dataset {dataset_choice}\n")
    

if __name__ == "__main__":
    # creating choice dictionary
    choice_dict = {}

    net_choices = np.array(['vgg', 'densenet', 'dla'])
    dataset_choices = np.array(['cifar10', 'MNIST'])

    choice_dict['net'] = net_choices
    choice_dict['dataset'] = dataset_choices
    create_makefile(choice_dict)
    
