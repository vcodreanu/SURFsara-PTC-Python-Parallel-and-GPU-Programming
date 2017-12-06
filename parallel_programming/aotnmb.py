import numpy as np
from numba.pycc import CC

cc = CC('SPD_MODULE')

@cc.export('single_path_diff_aot', 'f8(f8, f8, f8, i8)')
def single_path_diff_aot(alpha, beta, sig, T):
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

cc.compile()

# Call function
import SPD_MODULE
import time
st=time.time()
SPD_MODULE.single_path_diff_aot(0.9, 0.0, 0.1, 100000)

print("AOT run {}s".format(time.time()-st))