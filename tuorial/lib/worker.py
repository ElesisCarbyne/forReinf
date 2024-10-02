import numpy as np

def square(x):
    return np.square(x)

def square2(i, x, queue):
    print("In process {}".format(i,))
    queue.put(np.square(x))