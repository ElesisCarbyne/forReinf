import numpy as np
import torch

def square(x):
    return np.square(x)

def square2(i, x, queue):
    print("In process {}".format(i,))
    queue.put(np.square(x))

def square3(i, x, queue):
    # temp = torch.pow(x, 2)
    temp = x + 10
    print("In process {}: {}".format(i, temp))
    return queue.put(temp)

def square4(i, x, queue):
    print("In process {}: {}".format(i, x))
    return queue.put(x)