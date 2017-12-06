#!/usr/bin/env python 

"""
Serial version of Matrix-Vector Multiplication.
This code will run *iter* iterations of
  v(t+1) = M * v(t)
where v is a vector of length *size* and M a dense size*size
matrix. 
"""

import numpy as np
from numpy.fft import fft2, ifft2
from math import ceil, fabs
import time

size = 10000          # lengt of vector v
iter = 50             # number of iterations to run

# This is the complete vector
vec = np.zeros(size)            # Every element zero...
vec[0] = 1.0                    #  ... besides vec[0]

mat =np.zeros([size, size] , dtype='f')
mat[:,0] = 1.0
start = time.time()

for t in range(iter):
  result = np.inner(mat, vec)

stop = time.time()
elapsed = stop - start    ### Stop stopwatch ###

if fabs(result[iter]-1.0) > 0.01:
    print("!! Error: Wrong result!")

print(" %d iterations of size %d in %5.2fs: %5.2f iterations per second" %
    (iter, size, elapsed, iter/elapsed) 
)
print("============================================================================")

