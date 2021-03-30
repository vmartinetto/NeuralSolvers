import torch
import numpy as np
import scipy

if __name__ == "__main__":

    # load data and make necessary vectors

    data = np.loadtxt('harmonic_oscillator_grnd.txt')
    xl = data[:,1]
    ef = data[:,2]
    xl = torch.from_numpy(xl)
    xl.requires_grad = True
    ef = torch.from_numpy(ef)
    normx = torch.empty(len(xl))
    for i,x in enumerate(xl):
        normx[i] = 1/x

    # multiply vectors to create graph

    norm = xl*normx
    ef_x = ef*norm

    # backward prop

    ones = torch.ones(len(ef_x))
    ef_x.backward(ones)
    print(xl.grad)