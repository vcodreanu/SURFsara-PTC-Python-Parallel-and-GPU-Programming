import numba
import numpy as np
import numpy.core.umath_tests as ut
import timeit
@numba.guvectorize(['void(float32[:,:], float32[:,:], float32[:,:])'],
                      '(m, n),(n, p)->(m, p)', target='cuda')
def batch_matrix_mult(a, b, c):
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            tmp = 0
            for n in range(a.shape[1]):
                 tmp += a[i, n] * b[n, j]
            c[i, j] = tmp


n = 4000000
dim = 2
a = np.random.random(n * dim * dim).astype(np.float32).reshape(n, dim, dim)
b = np.random.random(n * dim * dim).astype(np.float32).reshape(n, dim, dim)

testcode_gpu = ''' 
def test_gpu(): 
    return batch_matrix_mult(a,b)
'''

print('Numba GPU time')
starttime = timeit.default_timer()
print("The start time is :",starttime)
for i in range(7):
    gpu_ans = batch_matrix_mult(a,b)
print("The time difference is :", timeit.default_timer() - starttime)
