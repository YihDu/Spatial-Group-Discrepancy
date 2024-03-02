import numpy as np
import torch
import ot
import time

def gaussian_emd(x, y, sigma=1e-3):
    X = np.stack(x)
    Y = np.stack(y)
    
    n = len(x)
    m = len(y)
    a = np.ones((n,)) / n
    b = np.ones((m,)) / m

    cost_matrix = ot.dist(X, Y, metric='euclidean')
    emd = ot.emd2(a, b, cost_matrix , numItermax=100000000)

    return np.exp(-emd * emd / (2 * sigma * sigma))

def disc(samples1, samples2, kernel, *args, **kwargs):
    d = 0
    n = len(samples1)
    m = len(samples2)
    
    loop_start_time = time.time()
    
    kernel_times = []

    for i in range(n):
        for j in range(m):
            kernel_start_time = time.time()
            s1 = samples1[i] 
            s2 = samples2[j]   
            d += kernel(s1, s2, *args, **kwargs)
            kernel_end_time = time.time()
            kernel_times.append(kernel_end_time - kernel_start_time)
    
    loop_end_time = time.time()
    print(f"Entire loop took {loop_end_time - loop_start_time:.5f} seconds.")
    print(f"Average kernel execution time: {sum(kernel_times) / len(kernel_times):.5f} seconds.")
    
    d /= n * m
    return d

def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    if is_hist:
        samples1 = [s1 / np.sum(s1) for s1 in samples1]
        samples2 = [s2 / np.sum(s2) for s2 in samples2]

    return disc(samples1, samples1, kernel, *args, **kwargs) + \
            disc(samples2, samples2, kernel, *args, **kwargs) - \
            2 * disc(samples1, samples2, kernel, *args, **kwargs)
            

