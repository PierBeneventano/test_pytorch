import os
import numpy as np
cwd = os.getcwd()   # get the current working directory

def create_makefile(choices_array):
    f = open(f"{cwd}/makefile","w")
    f.write(".PHONY = help setup test run clean environment\n\n")
    f.write(".DEFAULT_GOAL = setup\n")
    f.write("setup:\n")
    for choice in choices_array:
        f.write(f"\t@python main.py --net {choice}\n")
    

if __name__ == "__main__":
    choices_array = np.array(['vgg', 'densenet', 'dla'])
    create_makefile(choices_array)
    
