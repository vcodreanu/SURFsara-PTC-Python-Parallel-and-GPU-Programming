#!/usr/bin/env python
# coding: utf-8

# # Some more NumbaPro examples

# # Nbody

# In[1]:


import numpy as np
from numba import jit, cuda, float32, float64, f4, f8
from timeit import default_timer as timer

dtype = np.float64
n = 16384
blksize = 128
eps = 1.e-2


# # 1st version - Numpy

# In[2]:


def induced_velocity(x, xvort, gam, vel):
    vel[:] = 0.
    for xv, g in zip(xvort, gam):
        r = x - xv
        rsq = np.sum(r * r, 1) + eps**2
        vel += g * np.transpose(np.array([r[:,1], -r[:,0]])) / rsq[:, np.newaxis]


# # 2nd version - Numba on the CPU

# In[3]:


def induced_velocity3(x, xvort, gam, vel):
    nx = x.shape[0]
    nvort = xvort.shape[0]
    for i in range(nx):
        vel[i,0] = 0.
        vel[i,1] = 0.
        for j in range(nvort):
            rsq = (x[i,0]-xvort[j,0])**2 + (x[i,1]-xvort[j,1])**2 + eps**2
            vel[i,0] += gam[j] * (x[i,1]-xvort[j,1]) / rsq
            vel[i,1] += -gam[j] * (x[i,0]-xvort[j,0]) / rs
            

@jit(target='cpu')
def induced_velocity2(x, xvort, gam, vel):
    nx = x.shape[0]
    nvort = xvort.shape[0]
    for i in range(nx):
        vel[i,0] = 0.
        vel[i,1] = 0.
        for j in range(nvort):
            rsq = (x[i,0]-xvort[j,0])**2 + (x[i,1]-xvort[j,1])**2 + eps**2
            vel[i,0] += gam[j] * (x[i,1]-xvort[j,1]) / rsq
            vel[i,1] += -gam[j] * (x[i,0]-xvort[j,0]) / rsq




# # 3rd version - Numba on the GPU (global memory)

# In[4]:


@cuda.jit('f8[:,:], f8[:,:], f8[:], f8[:,:]')
def induced_velocity3(x, xvort, gam, vel):
    # eps = float32(1.e-2)
    # i, j = cuda.grid(2)
    i = cuda.grid(1)
    if i < x.shape[0]:
        vel[i,0] = float32(0.)
        vel[i,1] = float32(0.)
        nvort = xvort.shape[0]
        for j in range(nvort):
            rsq = (x[i,0]-xvort[j,0])**2 + (x[i,1]-xvort[j,1])**2 + eps**2
            vel[i,0] += gam[j] * (x[i,1]-xvort[j,1]) / rsq
            vel[i,1] += -gam[j] * (x[i,0]-xvort[j,0]) / rsq


# # 4th version - Numba on the GPU (shared memory)

# In[5]:


@cuda.jit('f8[:,:], f8[:,:], f8[:], f8[:,:]')
def induced_velocity4(x, xvort, gam, vel):
    smem = cuda.shared.array((blksize, 3), dtype=f8)
    t = cuda.threadIdx.x
    i = cuda.grid(1)
    # eps = 1.e-2
    nvort = xvort.shape[0]
    nx = x.shape[0]
    if i < nx:
        x0 = x[i,0]
        x1 = x[i,1]
    xvel = 0
    yvel = 0
    nvort = xvort.shape[0]
    for blk in range((nvort - 1) // blksize + 1):
        # load vortex positions and strengths into shared memory
        j = blk * blksize + t
        if j < nvort:
            smem[t,0] = xvort[j,0]
            smem[t,1] = xvort[j,1]
            smem[t,2] = gam[j]
        else:
            smem[t,0] = 0
            smem[t,1] = 0
            smem[t,2] = 0
        cuda.syncthreads()

        # compute the contributions to the velocity
        for k in range(blksize):
            rsq = (x0-smem[k,0])**2 + (x1-smem[k,1])**2 + eps**2
            xvel +=  smem[k,2] * (x1-smem[k,1]) / rsq
            yvel += -smem[k,2] * (x0-smem[k,0]) / rsq
        cuda.syncthreads()
    if i < nx:
        vel[i,0] = xvel
        vel[i,1] = yvel


# # And now, let's benchmark the three implementations

# In[6]:


def main():
    vort = np.array(np.random.rand(2*n), dtype=dtype).reshape((n,2))
    gamma = np.array(np.random.rand(n), dtype=dtype)
    vel = np.zeros_like(vort)
    start = timer()
    induced_velocity(vort, vort, gamma, vel)
    numpy_time = timer() - start
    print("n = %d" % n)
    print("Numpy".center(40, "="))
    print("Time: %f seconds" % numpy_time)

    vel2 = np.zeros_like(vort)
    start = timer()
    induced_velocity2(vort, vort, gamma, vel2)
    numba_time = timer() - start
    print("Numba".center(40, "="))
    print("Time: %f seconds" % numba_time)
    error = np.max(np.max(np.abs(vel2 - vel)))
    print("Difference: %f" % error)
    print("Speedup: %f" % (numpy_time / numba_time))

    stream = cuda.stream()
    d_vort = cuda.to_device(vort, stream)
    d_gamma = cuda.to_device(gamma, stream)
    vel3 = np.zeros_like(vort)
    d_vel = cuda.to_device(vel3, stream)
    # blockdim = (32,32)
    # griddim = (n // blockdim[0], n // blockdim[1])
    griddim = (n - 1) // blksize + 1
    start = timer()
    induced_velocity3[griddim, blksize, stream](d_vort, d_vort, d_gamma, d_vel)
    d_vel.to_host(stream)
    gpu_time = timer() - start
    error = np.max(np.max(np.abs(vel3 - vel)))
    print("GPU".center(40, "="))
    print("Time: %f seconds" % gpu_time)
    print("Difference: %f" % error)
    print("Speedup: %f" % (numpy_time / gpu_time))
    # print(vel3)

    vel4 = np.zeros_like(vort)
    d_vel2 = cuda.to_device(vel4, stream)
    start = timer()
    induced_velocity4[griddim, blksize, stream](d_vort, d_vort, d_gamma, d_vel2)
    d_vel2.to_host(stream)
    gpu2_time = timer() - start
    error = np.max(np.max(np.abs(vel4 - vel)))
    print("GPU smem".center(40, "="))
    print("Time: %f seconds" % gpu2_time)
    print("Difference: %f" % error)
    print("Speedup: %f" % (numpy_time / gpu2_time))
    # print("Expected".center(40,'-'))
    # print(vel)
    # print("GPU".center(40,'-'))
    # print(vel4)
    # print("Difference".center(40,'-'))
    # print(vel4 - vel)

if __name__ == "__main__":
    main()

