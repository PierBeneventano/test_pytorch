import os
import numpy as np
cwd = os.getcwd()   # get the current working directory

def create_makefile(choice_dict):
    f = open(f"{cwd}/makefile","w")
    f.write(".PHONY = help setup test run clean environment\n\n")
    f.write(".DEFAULT_GOAL = setup\n")
    f.write("setup:\n")
    for ln_choice in choice_dict['label_noise_prob']:
        if ln_choice != 0:
            for ln_sched_choice in choice_dict['ln_sched']:
                f.write(f"\t@python main.py --dataset 'MNIST' --label_noise {ln_choice} --ln_sched {ln_sched_choice}\n")
                for net_choice in choice_dict['net']:
                    f.write(f"\t@python main.py --net {net_choice} --label_noise {ln_choice} --ln_sched {ln_sched_choice}\n")
    

if __name__ == "__main__":
    # creating choice dictionary
    choice_dict = {}

    net_choices = np.array(['vgg', 'densenet', 'dla'])
    label_noise_choices = np.array([0.5, 0.2, 0.1, 0.0])
    ln_sched_choices = np.array(['decay', 'fixed'])

    choice_dict['net'] = net_choices
    choice_dict['label_noise_prob'] = label_noise_choices
    choice_dict['ln_sched'] = ln_sched_choices

    create_makefile(choice_dict)
    
