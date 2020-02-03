#!/usr/bin/env python
# coding: utf-8

# # Introduction to Python GPU Programming with Numba

# In[ ]:


# Misc. import
from __future__ import print_function
from IPython.display import Image 
#get_ipython().run_line_magic('matplotlib', 'inline')


# # Numba
# * Opensource BSD license
# * Basic CUDA GPU JIT compilation
# * OpenCL support coming

# In[ ]:


import numba
print("numba", numba.__version__)


# # The CUDA GPU
# 
# - A massively parallel processor (many cores)
#     - 100 threads, 1,000 threads, and more
# - optimized for data throughput
#     - Simple (shallow) cache hierarchy
#     - Best with manual caching!
#     - Cache memory is called shared memory and it is addressable
# - CPU is latency optimized
#     - Deep cache hierarchy
#     - L1, L2, L3 caches
# - GPU execution model is different
# - GPU forces you to think and program *in parallel*

# In[ ]:


# Get all the imports we need
import numba.cuda
import numpy as np
import math

my_gpu = numba.cuda.get_current_device()
print("Running on GPU:", my_gpu.name)
cores_per_capability = {
    1: 8,
    2: 32,
    3: 192,
    5: 128,
    7: 64
}
cc = my_gpu.compute_capability
print("Compute capability: ", "%d.%d" % cc, "(Numba requires >= 2.0)")
majorcc = cc[0]
print("Number of streaming multiprocessor:", my_gpu.MULTIPROCESSOR_COUNT)
cores_per_multiprocessor = cores_per_capability[majorcc]
print("Number of cores per mutliprocessor:", cores_per_multiprocessor)
total_cores = cores_per_multiprocessor * my_gpu.MULTIPROCESSOR_COUNT
print("Number of cores on GPU:", total_cores)


# # High-level Array-Oriented Style
# 
# - Use NumPy array as a unit of computation
# - Use NumPy universal function (ufunc) as an abstraction of computation and scheduling
# - ufuncs are elementwise functions
# - If you use NumPy, you are using ufuncs

# In[ ]:


print(np.sin, "is of type", type(np.sin))
print(np.add, "is of type", type(np.add))


# # Vectorize
# 
# - generate a ufunc from a python function
# - converts scalar function to elementwise array function
# - Numba provides CPU and GPU support

# In[ ]:


# CPU version
@numba.vectorize(['float32(float32, float32)',
                  'float64(float64, float64)'], target='cpu')
def cpu_sincos(x, y):
    return math.sin(x) * math.cos(y)

# CUDA version
@numba.vectorize(['float32(float32, float32)',
                     'float64(float64, float64)'], target='cuda')
def gpu_sincos(x, y):
    return math.sin(x) * math.cos(y)


# ```
# @numba.vectorize(<list of signatures>, target=<'cpu', 'cuda'>)
# ```
# 
# - A ufunc can be overloaded to work on multiple type signatures
# 

# ### Test it out
# 
# - 2 input arrays
# - 1 output array
# - 1 million doubles (8 MB) per array
# - Total 24 MB of data

# In[ ]:


# Generate data
n = 1000000
x = np.linspace(0, np.pi, n)
y = np.linspace(0, np.pi, n)

# Check result
np_ans = np.sin(x) * np.cos(y)
nb_cpu_ans = cpu_sincos(x, y)
nb_gpu_ans = gpu_sincos(x, y)

print("CPU vectorize correct: ", np.allclose(nb_cpu_ans, np_ans))
print("GPU vectorize correct: ", np.allclose(nb_gpu_ans, np_ans))


# ## Benchmark

# In[ ]:


print("NumPy")
get_ipython().run_line_magic('timeit', 'np.sin(x) * np.cos(y)')

print("CPU vectorize")
get_ipython().run_line_magic('timeit', 'cpu_sincos(x, y)')

print("GPU vectorize")
get_ipython().run_line_magic('timeit', 'gpu_sincos(x, y)')

# Optional cleanup 
del x, y


# - CPU vectorize time is similar to pure NumPy time because ``sin()`` and ``cos()`` calls dominate the time.
# - GPU vectorize is a lot faster
# 

# 
# ## Behind the scence
# 
# 

# 
# ### Automatic memory transfer
# 
# - NumPy arrays are automatically transferred
#     - CPU -> GPU
#     - GPU -> CPU

# 
# ### Automatic work scheduling
# 
# - The work is distributed the across all threads on the GPU
# - The GPU hardware handles the scheduling
# 

# 
# ### Automatic GPU memory management
# 
# - GPU memory is tied to object lifetime
# - freed automatically

# # Generalized Universal Function (guvectorize)
# 
# - Vectorize is limited to scalar arguments in the core function
# - GUVectorize accepts array arguments

# In[ ]:


@numba.guvectorize(['void(float32[:,:], float32[:,:], float32[:,:])'],
                      '(m, n),(n, p)->(m, p)', target='cuda')
def batch_matrix_mult(a, b, c):
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            tmp = 0
            for n in range(a.shape[1]):
                 tmp += a[i, n] * b[n, j]
            c[i, j] = tmp


# ```
# @numba.guvectorize(<list of function signatures>, <a string to desc the shape signature>, target=<'cpu', 'cuda'>)
# ```

# In[ ]:


a = np.arange(1.0, 10.0, dtype=np.float32).reshape(3,3)
b = np.arange(1.0, 10.0, dtype=np.float32).reshape(3,3)

# Use the builtin matrix_multiply in NumPy for CPU test
import numpy.core.umath_tests as ut

# Check result
print('NumPy result')
np_ans = ut.matrix_multiply(a, b)
print(np_ans)

print('Numba GPU result')
gpu_ans = batch_matrix_mult(a, b)
print(gpu_ans)

assert np.allclose(np_ans, gpu_ans)


# ### Test it out
# 
# - batch multiply 4 million 2x2 matrices 

# In[ ]:


n = 4000000
dim = 2
a = np.random.random(n * dim * dim).astype(np.float32).reshape(n, dim, dim)
b = np.random.random(n * dim * dim).astype(np.float32).reshape(n, dim, dim)

print('NumPy time')
get_ipython().run_line_magic('timeit', 'ut.matrix_multiply(a, b)')

print('Numba GPU time')
get_ipython().run_line_magic('timeit', 'batch_matrix_mult(a, b)')


# - GPU time seems to be similar to CPU time
# 

# 
# ## Manually Transfer the data to the GPU
# 
# - This will let us see the actual compute time without the CPU<->GPU transfer

# In[ ]:


dc = numba.cuda.device_array_like(a)
da = numba.cuda.to_device(a)
db = numba.cuda.to_device(b)


# * ```numba.cuda.device_array_like``` allocate without initialization with the type and shape of another array.
#     * similar to ```numpy.empty_like(a)```
# * ```numba.cuda.to_device``` create a GPU copy of the CPU array
# 

# 
# ## Pure compute time on the GPU

# In[ ]:


def check_pure_compute_time(da, db, dc):
    batch_matrix_mult(da, db, out=dc)
    numba.cuda.synchronize()   # ensure the call has completed
    
get_ipython().run_line_magic('timeit', 'check_pure_compute_time(da, db, dc)')
del da, db, dc


# * Actual compute time is **a lot faster**
# * PCI-express transfer overhead

# #### Tips
# If you have a sequence of ufuncs to apply, pin the data on the GPU by manual transfer

# -----
# 
# # Low-Level Approach: @numba.cuda.jit
# 
# - Numba can generate CUDA functions with the `@jit` decorator
# - Decorated function follows CUDA execution model 

# ## CUDA Execution Model
# 
# - Kernel functions
#     - visible to the host CPU
#     - cannot return any value
#         - use output argument
#     - associates to a _grid_
# - Grid
#     - a group of blocks
#     - 1D, 2D, 3D
# - Blocks
#     - a group of threads
#     - 1D, 2D, 3D  
# - Every thread executes the same kernel
#     - thread can use the grid, block, thread coordinate system to determine its ID

# In[ ]:


Image(url='http://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/grid-of-thread-blocks.png')


# ## Compiling a CUDA Kernel

# In[ ]:


from numba import cuda

@numba.cuda.jit("void(float32[:], float32[:], float32[:])")
def vadd(arr_a, arr_b, arr_out):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x    # number of threads per block
    i = tx + bx * bw
    if i >= arr_out.size:
        return
    arr_out[i] = arr_a[i] + arr_b[i]


# ### Code Explained
# 
# #### Define a CUDA kernel with three 1D float32 arrays as args
# 
# ```
# @numba.cuda.jit("void(float32[:], float32[:], float32[:])")
# def vadd(arr_a, arr_b, arr_out):
# ```
# 
# #### Map thread, block coordinate to global position
# ```
#     tx = cuda.threadIdx.x   # thread label (along x dimension)
#     bx = cuda.blockIdx.x    # block label (along x dimension)
#     bw = cuda.blockDim.x    # number of threads in each block (along x dimension)
#     i = tx + bx * bw        # flattened linear address for each thread
# ```
# or simplified to:
# ```
#     i = cuda.grid(1)
# ``` 
# 
# #### Ensure global position is within array size
# 
# ```
#     if i >= arr_out.size:
#         return
# ```
# 
# #### The actual work
# 
# ```
#     arr_out[i] = arr_a[i] + arr_b[i]
# ```

# ## Launch kernel

# #### Prepare data

# In[ ]:


n = 100
a = np.arange(n, dtype=np.float32)
b = np.arange(n, dtype=np.float32)
c = np.empty_like(a)                 # Must prepare the output array to hold the result


# #### Calculate thread, block count
# 
# - thread count is set to **warp size** of the GPU
#     - Warp size is similar to SIMD vector width on the CPU
#     - **performance tips**: set thread count to multiple of warp size
# - block count is ceil(n/thread_ct)
# 
# **Note:**
# This will launch more threads than there are elements in the array

# In[ ]:


thread_ct = my_gpu.WARP_SIZE
block_ct = int(math.ceil(float(n) / thread_ct))

print("Threads per block:", thread_ct)
print("Block per grid:", block_ct)


# #### Launch kernel
# 
# Kernel function object uses the ``__getitem__`` (indexing notation) to configure the grid and block dimensions.
# 
# ```
#     kernel_function[griddim, blockdim](*args)
# ```
# 
# - **griddim**
#     - Number of blocks per grid (grid dimension)
#     - type: int for 1d or 1,2,3-tuple of ints for 1d, 2d, or 3d respectively
# 
# - **blockdim**: 
#     - Number of threads per block (blockdim dimension)
#     - type: int for 1d or 1,2,3-tuple of ints for 1d, 2d, or 3d respectively
# 

# In[ ]:


vadd[block_ct, thread_ct](a, b, c)    # Last argument is the output array in this case
print(c)


# # Example: Matrix Matrix Multiplication
# 
# - Show manual caching with shared memory
# - Not the best matrix matrix multiplication implementation

# #### Prepare constants

# In[ ]:


from numba import float32

bpg = 150
tpb = 32
n = bpg * tpb
shared_mem_size = (tpb, tpb)
griddim = bpg, bpg
blockdim = tpb, tpb


# #### Naive version

# In[ ]:


Image(url="http://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/matrix-multiplication-without-shared-memory.png")


# In[ ]:


@numba.cuda.jit("void(float32[:,:], float32[:,:], float32[:,:])")
def naive_matrix_mult(A, B, C):
    x, y = cuda.grid(2)
    if x >= n or y >= n:
        return

    C[y, x] = 0
    for i in range(n):
        C[y, x] += A[y, i] * B[i, x]


# #### Optimized version (shared memory + blocking)

# In[ ]:


Image(url="http://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/matrix-multiplication-with-shared-memory.png")


# In[ ]:


@numba.cuda.jit("void(float32[:,:], float32[:,:], float32[:,:])")
def optimized_matrix_mult(A, B, C):
    # Declare shared memory
    sA = cuda.shared.array(shape=shared_mem_size, dtype=float32)
    sB = cuda.shared.array(shape=shared_mem_size, dtype=float32)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    x, y = cuda.grid(2)

    acc = 0
    for i in range(bpg):
        if x < n and y < n:
            # Prefill cache
            sA[ty, tx] = A[y, tx + i * tpb]
            sB[ty, tx] = B[ty + i * tpb, x]

        # Synchronize all threads in the block
        cuda.syncthreads()

        if x < n and y < n:
            # Compute product
            for j in range(tpb):
                acc += sA[ty, j] * sB[j, tx]

        # Wait until all threads finish the computation
        cuda.syncthreads()

    if x < n and y < n:
        C[y, x] = acc


# #### Prepare data

# In[ ]:


# Prepare data on the CPU
A = np.array(np.random.random((n, n)), dtype=np.float32)
B = np.array(np.random.random((n, n)), dtype=np.float32)

print("N = %d x %d" % (n, n))

# Prepare data on the GPU
dA = cuda.to_device(A)
dB = cuda.to_device(B)
dC = cuda.device_array_like(A)


# #### Benchmark

# In[ ]:


# Time the unoptimized version
import time
s = time.time()
naive_matrix_mult[griddim, blockdim](dA, dB, dC)
numba.cuda.synchronize()
e = time.time()
unopt_ans = dC.copy_to_host()
tcuda_unopt = e - s

# Time the optimized version
s = time.time()
optimized_matrix_mult[griddim, blockdim](dA, dB, dC)
numba.cuda.synchronize()
e = time.time()
opt_ans = dC.copy_to_host()
tcuda_opt = e - s


# #### Result

# In[ ]:


assert np.allclose(unopt_ans, opt_ans)
print("Without shared memory:", "%.2f" % tcuda_unopt, "s")
print("With shared memory:", "%.2f" % tcuda_opt, "s")


# # Summary
# 
# - Numba
#     - opensource low-level GPU support
#     - CUDA kernel ``@numba.cuda.jit``
#     - vectorize ``@numba.vectorize``
#     - guvectorize ``@numba.guvectorize``
# 

# In[ ]:




