import numpy as np
from numba import jit

def single_path_diff(alpha, beta, sig, T):
    # Fill first element with mean
    x = beta / (1-alpha)
    y = x

    # Various variables we need
    xi_yi = 0
    xi = 0
    yi = 0
    xi2 = 0

    for i in range(T):
        # Update x
        x = beta + alpha*y + sig*np.random.randn()

        # Update all variables
        xi_yi += x*y
        xi += x
        yi += y
        xi2 += x*x

        # Update y
        y = x

    spd = (xi_yi - (1/T)*xi*yi)/(xi2 - (1/T)*xi*xi) - alpha
    return spd

single_path_diff_jit = jit(single_path_diff, nopython=False, cache=True)
import time
st=time.time()
single_path_diff(0.9, 0.0, 0.1, 100000)
print("Vanilla run {}s".format(time.time()-st))
st = time.time()
single_path_diff_jit(0.9, 0.0, 0.1, 100000)
print("Jit run {}s".format(time.time()-st))