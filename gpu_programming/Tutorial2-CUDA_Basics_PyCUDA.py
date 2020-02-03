#!/usr/bin/env python
# coding: utf-8

# # Introduction to Jupyter (JUlia, PYThon, and R), bash, and CUDA

# ### We can we both text and code in Jupyter, but first let's start with some bash

# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('cat job.jupyter.gpu')


# In[ ]:


get_ipython().system('nvcc --version')


# ### Jupyter has some "magic" builtin commands. Here's how to list them

# In[ ]:


get_ipython().run_line_magic('lsmagic', '')


# And now let's use the bash cell magic

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'echo "Hello 1"\nsleep 5\necho "Hello 2"')


# Here's how to get some documentation for the magics

# In[ ]:


get_ipython().set_next_input('% autosave');get_ipython().run_line_magic('pinfo', 'autosave')


# And the "magic" source code

# In[ ]:


get_ipython().set_next_input('% autosave');get_ipython().run_line_magic('pinfo2', 'autosave')


# And here's how we get some documentation on Python functions.

# In[ ]:


get_ipython().run_line_magic('pinfo', 'str.replace')


# Or on important Python packages such as numpy

# In[ ]:


import numpy
get_ipython().run_line_magic('pinfo', 'numpy')


# One can list the currently set environment

# In[ ]:


get_ipython().run_line_magic('env', '')


# Or modify it

# In[ ]:


get_ipython().run_line_magic('env', 'OMP_NUM_THREADS=16')


# That's how we load a Python program

# In[ ]:


# %load ./hello_world.py
# After Running
# %load ./hello_world.py
if __name__ == "__main__":
	print("Hello World!")


# And that's how we run it

# In[ ]:


get_ipython().run_line_magic('run', './hello_world.py')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'for i in a b c;\ndo\necho $i\ndone')


# # PyCUDA basics.

# In[ ]:


import pycuda
import numpy as np


# ## Exploring your GPU device(s).

# Listing devices:

# In[ ]:


from pycuda import autoinit
from pycuda.tools import DeviceData


# In[ ]:

#%autocall

specs = DeviceData()


# In[ ]:


print ('Max threads per block = ',specs.max_threads)


# In[ ]:


print ('Warp size            =', specs.warp_size)
print ('Warps per MP         =', specs.warps_per_mp)
print ('Thread Blocks per MP =', specs.thread_blocks_per_mp)
print ('Registers            =', specs.registers)
print ('Shared memory        =', specs.shared_memory)
print ('Granularity ??       =', specs.smem_granularity)


# Another way to list devices

# In[ ]:


import pycuda.driver as drv


# In[ ]:


drv.init()


# In[ ]:


drv.get_version()


# In[ ]:


devn = drv.Device.count()
print ('Localized GPUs =',devn)


# In[ ]:


devices = []
for i in range(devn):
    devices.append(drv.Device(i))


# All you want to know about your GPU, but you're afraid to ask!

# In[ ]:


for sp in devices:
    print ('Name = ',sp.name())
    print ('PCI Bus = ',sp.pci_bus_id())
    print ('Compute Capability = ',sp.compute_capability())
    print ('Total Memory = ',sp.total_memory()/(2.**20) , 'MBytes')
    attr = sp.get_attributes()
    for j in range(len(list(attr.items()))):
        print (list(attr.items())[j])#,'Bytes (when apply)'
    print ('------------------')
    print ('------------------')


# MAX_THREADS_PER_BLOCK, 1024
# 
# For example for a 3D mesh (less optimal), we only have available $$8\times 8\times 8 = 512 \,simetric$$ 
#  $$8\times 8\times 16 = 1024 \,cilindrical$$
# block size per dimension = 8 or 16.
# In 2D case the optimal value is:
# $$32\times32 = 1024$$
# For the 1D case we has $$1024$$
# 
# 
# MAX_THREADS_PER_MULTIPROCESSOR, $2048 = 4*2^9$
# 
# If we can take this literally, we can process in one processor about 4 meshes of $8\times8\times8$, or four blocks of 3D meshes. With this result, we can evaluate the efficiency comparing cilindrical and symetric performance
# 

# ### Now your device has ..

# In[ ]:


drv.mem_get_info()[0]/(2.**20),'MB of Free Memory',drv.mem_get_info()[1]/(2.**20),'MB Total Memory'


# Let's think in array sizes. For example a float of 4 bytes length:

# In[ ]:


print ('Linear max length:', drv.mem_get_info()[0]/(4))
print ('2D max length    :', np.sqrt(drv.mem_get_info()[0]/(4)))
print ('3D max length    :', np.power(drv.mem_get_info()[0]/(4),1./3.))


# In[ ]:


get_ipython().system('nvidia-smi')


# # __CUDA__ __C__

# <a href="http://docs.nvidia.com/cuda"><img src="images/CUDA.png" width="30%" /></a>

# ## Basic example. Vector Addition

# > To solve the problem from the Figure below, we will present a basic C implementation. Subsequently, you can find a pure CUDA C implementation. As you will see below in order to program the GPU, an interplay between C (that does the bookkeeping) and CUDA (that does the heavylifting) is needed.
# 
# ![Alt text](images/suma.png)

# ### Version C

# ```c
# #include <stdio.h>
# 
# int main(void)
# {
# int N = 10;
# float a[N],b[N],c[N];
# 
# for (int i = 0; i < N; ++i){
# 	a[i] = i;
# 	b[i] = 2.0f;	
# }
# 
# for (int i = 0; i < N; ++i){
# 	c[i]= a[i]+b[i];	
# }
# 
# for (int i = 0; i < N; ++i){
# 	printf("%f \n",c[i]);	
# }
# 
# 
# return 0;
# }```

# In[ ]:


get_ipython().system('g++ cpuAdd.c -o cpua')


# In[ ]:


get_ipython().system('cat cpuAdd.c')


# In[ ]:


get_ipython().system('./cpua')


# ### Version CUDA C

# ![Alt text](images/cuda3.png)

# ![Alt text](images/CUDAmodelThreads.png)

# ```c
# #include <stdio.h>
# #include <cuda_runtime.h>
# // CUDA Kernel
# __global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
# {
#     int i = blockDim.x * blockIdx.x + threadIdx.x;
#     if (i < numElements)
#     {
#         C[i] = A[i] + B[i];
#     }
# }
# 
# /**
#  * Host main routine
#  */
# int main(void)
# {
#     int numElements = 15;
#     size_t size = numElements * sizeof(float);
#     printf("[Vector addition of %d elements]\n", numElements);
# 
#     float a[numElements],b[numElements],c[numElements];
#     float *a_gpu,*b_gpu,*c_gpu;
# 
#     cudaMalloc((void **)&a_gpu, size);
#     cudaMalloc((void **)&b_gpu, size);
#     cudaMalloc((void **)&c_gpu, size);
# 
#     for (int i=0;i<numElements;++i ){
#     
#     	a[i] = i*i;
#     	b[i] = i;
#     
#     }
#     // Copy the host input vectors A and B in host memory to the device input vectors in
#     // device memory
#     printf("Copy input data from the host memory to the CUDA device\n");
#     cudaMemcpy(a_gpu, a, size, cudaMemcpyHostToDevice);
#     cudaMemcpy(b_gpu, b, size, cudaMemcpyHostToDevice);
# 
#     // Launch the Vector Add CUDA Kernel
#     int threadsPerBlock = 256;
#     int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
#     printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
#     vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(a_gpu, b_gpu, c_gpu, numElements);
# 
#     // Copy the device result vector in device memory to the host result vector
#     // in host memory.
#     printf("Copy output data from the CUDA device to the host memory\n");
#     cudaMemcpy(c, c_gpu, size, cudaMemcpyDeviceToHost);
# 
#     for (int i=0;i<numElements;++i ){
#     	printf("%f \n",c[i]);
#     }
# 
#     // Free device global memory
#     cudaFree(a_gpu);
#     cudaFree(b_gpu);
#     cudaFree(c_gpu);
#     
#     printf("Done\n");
#     return 0;
# }
# ```

# In[ ]:


get_ipython().system('nvcc gpuAdd.cu -o gpu')


# In[ ]:


get_ipython().system('./gpu')


# In[ ]:


get_ipython().system('nvidia-smi')


# ### 1st PyCUDA implementation: GPUArrays

# In[ ]:


from pycuda import autoinit
from pycuda import gpuarray
import numpy as np


# In[ ]:


aux = range(15)
a = np.array(aux).astype(np.float32)
b = (a*a).astype(np.float32)
c = np.zeros(len(aux)).astype(np.float32)


# In[ ]:


a_gpu = gpuarray.to_gpu(a)


# In[ ]:


b_gpu = gpuarray.to_gpu(b)
c_gpu = gpuarray.to_gpu(c)


# In[ ]:


aux_gpu = a_gpu+b_gpu


# In[ ]:


aux_gpu.gpudata


# In[ ]:


type(aux_gpu)


# In[ ]:


a_gpu,b_gpu,aux_gpu


# ### 2nd PyCUDA implementation: Elementwise Kernels

# In[ ]:


from pycuda.elementwise import ElementwiseKernel


# In[ ]:


c_gpu.dtype


# In[ ]:


myCudaFunc = ElementwiseKernel(arguments = "float *a, float *b, float *c",
                               operation = "c[i] = a[i]+b[i]",
                               name = "mySumK")


# In[ ]:


myCudaFunc(a_gpu,b_gpu,c_gpu)


# In[ ]:


c_gpu


# ### 3rd PyCUDA implementation

# In[ ]:


from pycuda.compiler import SourceModule


# In[ ]:


cudaCode = open("gpuAdd.cu","r")
myCUDACode = cudaCode.read()


# In[ ]:


myCode = SourceModule(myCUDACode)


# In[ ]:


importedKernel = myCode.get_function("vectorAdd")


# In[ ]:


get_ipython().system('ls')


# In[ ]:


nData = len(a)


# In[ ]:


nData


# In[ ]:


nThreadsPerBlock = 256
nBlockPerGrid = 1
nGridsPerBlock = 1


# In[ ]:


c_gpu.set(c)


# In[ ]:


c_gpu


# In[ ]:


importedKernel(a_gpu.gpudata,b_gpu.gpudata,c_gpu.gpudata,block=(256,1,1))


# In[ ]:


c_gpu


# # Summary

# So far, we can summarize the following:
# 
# **Kernel**
# >The CUDA kernel is the elementary function of parallelization. It features an extended C syntax and is the unit of computation that runs in parallel on the thousands of cores that compose a GPU. The kernel can be of one of the following tyes.
# 
# > __global__ - denotes general CUDA kernel. These functions are called from the host
# 
# > __device__ - represents a device (GPU) function. - These functions can be called either from __device__ or __global__
# 
# > __host__ - represents a host (CPU) function.
# 
# **And how does parallelization work?**
# >Each time a kernel is called it is necessary to give it a thread distribution (or _threads_) which are organized in blocks (_blocks_) and these in turn in a _grid_ (these can have different dimensions: 1D, 2D, 3D). These threads are copies of the kernel and each is a process to be carried out on the GPU cores, i.e. if we launch a grid with 5 blocks (_gridDim_ = (5,1,1)) with 10 threads per block (_blockDim_ = (10,1,1)), then we will have launched 50 tasks in parallel. Although the kernels to be executed by the threads are copies of the one that we originally wrote, the differentiation is given by the assignment of a counter to each process. The usual way to determine this ** global process index ** is exemplified below:
# 
# ![Alt text](images/CUDAmodelThreads.png)
# 
# >(**NOTICE** This can change depending on the number of blocks and threads)
# ![Alt text](images/cuda-grid.png)
# 
# >For our vector sum example we have used the **global process index** so that each thread makes the sum over a different component of the vectors. It is at this point that the parallelization appears, since each thread operates on a different component of the vector.
# 
# 
# **PyCUDA**
# >This Python library lets you access Nvidia‘s CUDA parallel computation API from Python. It allows us in principle to do everything we can do with CUDA C, but in a simpler way. One of the virtues of PyCUDA is that is allows us to use the class **GPUArray**, which in turn allows us to easily manage memory, assign of values, perform data transfer between CPU and GPU, etc. This class of pyCUDA maintains the same structure as the **numpy** library, giving developers the same feel as they were using **numpy**.
# 
# >After initializing the context of pyCUDA we can make use of the class GPUArray. The simplest way to generate an array in the global memory of the GPU is through _gpuarray.to_gpu (), where the value that is passed to the function is a **numpy** array. Although all GPU global memory arrays are linear arrays, the GPUArray class handles the possibility of preserving array dimensions. 

# ### References

# #<a href="http://documen.tician.de/pycuda/">pyCUDA</a>
# 
# #<a href="http://docs.scipy.org/doc/numpy/reference/">Numpy</a>
# 
# #<a href="http://docs.nvidia.com/cuda">CUDA</a>

# # Matrix addition

# ![Alt text](images/cudaMatrix.png)

# In[ ]:


import numpy as np
from pycuda import gpuarray, autoinit
import pycuda.driver as cuda
from pycuda.tools import DeviceData
from pycuda.tools import OccupancyRecord as occupancy
nDevices = cuda.Device.count()
ndev = None
for i in range(nDevices):
	dev = cuda.Device( i )
	print ("  Device {0}: {1}".format( i, dev.name() ))
devNumber = 0
if nDevices > 1:
	if ndev == None:
	  devNumber = int(raw_input("Select device number: "))
	else:
	  devNumber = ndev
dev = cuda.Device( devNumber)
cuda.Context.pop()
ctxCUDA = dev.make_context()
devdata = DeviceData(dev)
print ("Using device {0}: {1}".format( devNumber, dev.name() ))


# Initialize the arrays in CPU memory using Numpy, the preffered array handling library.

# In[ ]:


presCPU, presGPU = np.float32, 'float'
#presCPU, presGPU = np.float64, 'double'


# In[ ]:


a_cpu = np.ones((512,512), dtype=presCPU)
a_cpu = np.random.random((512,512)).astype(presCPU)
b_cpu = np.ones((512,512), dtype=presCPU)
b_cpu = np.random.random((512,512)).astype(presCPU)
c_cpu = np.zeros((512,512), dtype=presCPU)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().run_line_magic('pinfo', 'np.random.random')


# In[ ]:


from matplotlib import pyplot as plt


# In[ ]:


plt.imshow(a_cpu)
plt.colorbar()


# In[ ]:


plt.imshow(b_cpu)
plt.colorbar()


# We now explicitly copy our arrays from the host (CPU) memory space to the device (GPU) memory space. These tasks are achieved using the GPUArray class from PyCUDA

# In[ ]:


a_gpu = gpuarray.to_gpu(a_cpu)
b_gpu = gpuarray.to_gpu(b_cpu)
c_gpu = gpuarray.to_gpu(c_cpu)
c_cpu


# We now perform the matrix summation on the CPU.

# In[ ]:


t_cpu = get_ipython().run_line_magic('timeit', '-o c_cpu = a_cpu+b_cpu ')
# notice the very nice %timeit function


# In[ ]:


c_cpu = a_cpu+b_cpu
c_cpu


# We now define the kernel (GPU function) that operates in a parallel manner on the two input matrices and generates the result in the output matrix C. Notice the C-like syntax.

# In[ ]:


cudaKernel = '''
__global__ void matrixAdd(float *A, float *B, float *C)
{
    int tid_x = blockDim.x * blockIdx.x + threadIdx.x;
    int tid_y = blockDim.y * blockIdx.y + threadIdx.y;
    int tid   = gridDim.x * blockDim.x * tid_y + tid_x;
    C[tid] = A[tid] + B[tid];
}
'''


# In[ ]:


print (cudaKernel)


# Now we will compile and generate a function from the previously written kernel. This is achieved in a simple way using pyCUDA.

# In[ ]:


from pycuda.compiler import SourceModule


# In[ ]:


myCode = SourceModule(cudaKernel)
addMatrix = myCode.get_function("matrixAdd") # The output of get_function is the GPU-compiled function.


# In[ ]:


type(addMatrix) 


# Now we must decide the distribution of blocks and threads appropriate to:
# > 1. Get the required number of tasks completed
# > 2. Choose an organization appropriate for the dimensions of the problem
# 
# 

# As we deal with matrices, ideally is to generate a 2D distribution of threads. This allows us to perform tasks on blocks of the matrices. NOTICE: If we wanted we could only reuse the vector sum code since in fact matrix summation is a special case of the vector summation exercise. 

# The choice for this case is more or less simple. Let $N$ be the dimension of the matrix. Since we have 1024 threads available per block, we can divide the array into blocks of $ \sqrt{1024} = 32 $. We will always try to exploit to the maximum the amount of thread per block when performing simple tasks. The grid will then be ($N/32$,$N/32$), thus (16,16,1).

# In[ ]:


cuBlock = (32,32,1) # In PyCUDA it is necessary to type in the 3rd grid dimension as well.
cuGrid = (16,16,1)
nthreads = cuBlock[0]*cuBlock[1]*cuBlock[2]
nthreads


# For hardware efficiency reasons, we will usually choose dimensions that are multiples of 32 or powers of 2. This is because the blocks are divided into warps (32-thread unit of execution), and these are executed in parallel in the multiprocessors (SM), so if we do not choose these dimensions wisely we might waste precious computation.

# Once we have set the work distribution for our kernel, we have two ways to invoke the GPU kernel. The first is through the direct use of the compiled function::
# ```python
# kernelFunction(arg1,arg2, ... ,block=(n,m,l),grid=(r,s,t))
# ```
# The second is through an intermediate step called "preparation" :
# ```python
# kernelFunction.prepare('ABC..') # Each letter corresponds to an input data type of the function, i = int, f = float, P = pointer, ...
# kernelFunction.prepared_call(grid,block,arg1.gpudata,arg2,...) # When using GPU arrays, they should be passed as pointers with the attribute 'gpudata'
# ```

# Simple usage

# In[ ]:


addMatrix(a_gpu,b_gpu,c_gpu,block=cuBlock,grid=cuGrid)


# Usage with "preparation"

# In[ ]:


addMatrix.prepare('PPP')


# In[ ]:


addMatrix.prepared_call(cuGrid,cuBlock,a_gpu.gpudata,b_gpu.gpudata,c_gpu.gpudata)


# In the "preparation" way, it is possible to measure the kernel's execution time

# In[ ]:


time2 = addMatrix.prepared_timed_call(cuGrid,cuBlock,a_gpu.gpudata,b_gpu.gpudata,c_gpu.gpudata)


# In[ ]:


time2()


# In[ ]:


c = c_gpu.get()  #Here we copy the GPU array back to CPU memory
c, c_cpu


# In[ ]:


plt.imshow(c-c_cpu,interpolation='none')
plt.colorbar()


# In[ ]:


np.sum(np.sum(np.abs(c_cpu-c)))


# In[ ]:


def getKernelInfo(kernel,nthreads, rt=True):
    ''' This function returns info about kernels theoretical performance, but warning is not trivial to optimize! '''
    shared=kernel.shared_size_bytes
    regs=kernel.num_regs
    local=kernel.local_size_bytes
    const=kernel.const_size_bytes
    mbpt=kernel.max_threads_per_block
    #threads =  #self.block_size_x* self.block_size_y* self.block_size_z
    occupy = occupancy(devdata, nthreads, shared_mem=shared, registers=regs)
    print ("==Kernel Memory==")
    print("""Local:        {0}
Shared:       {1}
Registers:    {2}
Const:        {3}
Max Threads/B:{4}""".format(local,shared,regs,const,mbpt))
    print ("==Occupancy==")
    print("""Blocks executed by SM: {0}
Limited by:            {1}
Warps executed by SM:  {2}
Occupancy:             {3}""".format(occupy.tb_per_mp,occupy.limited_by,occupy.warps_per_mp,occupy.occupancy))
    if rt:
        return occupy.occupancy
    
def gpuMesureTime(myKernel, ntimes=1000):
    start = cuda.Event()
    end = cuda.Event()
    start.record()
    for i in range(ntimes):
      myKernel()
    end.record()
    end.synchronize()
    timeGPU = start.time_till(end)*1e-3
    print ("Call the function {0} times takes in GPU {1} seconds.\n".format(ntimes,timeGPU))
    print ("{0} seconds per call".format(timeGPU/ntimes))
    return timeGPU


# We can evaluate the performance of our kernel as well as the way in which the calculation is distributed per block.

# In[ ]:


getKernelInfo(addMatrix,nthreads)


# The getKernelInfo function provides us with a useful overview of memory usage and occupancy, which may help us to optimize the kernel.
# 
# Kernel memory: for example, comparing the number of used registers to the total registers available on your hardware gives you an idea of how many more your kernel could use.
# 
# Occupancy: higher occupancy can help to speed up calculations by hiding memory latency. Global memory in a GPU has a high latency, but when one thread is waiting for memory, another thread can already be started. For this to work efficiently, a sufficiently large number of threads should be executed by the SM at any given time. The number of threads scheduled is the warps executed by SM times the warp size. The maximum number of warps on an NVidia K40 is 48, hence, with a kernel launching 32 warps per SM, your occupancy is 0.67. It strongly depends on the kernel how high the occupancy needs to be to hide latency completely.

# In[ ]:


timeGPU=[]


# In[ ]:


get_ipython().run_line_magic('time', 'for i in range(1000): timeGPU.append(addMatrix.prepared_timed_call(cuGrid,cuBlock,a_gpu.gpudata,b_gpu.gpudata,c_gpu.gpudata)())')


# In[ ]:


np.sum(np.array(timeGPU))*1000


# # Matrix multiplication

# ![Alt text](images/matrixMul.png)

# In the matrix multiplication case, each thread will calculate an input of the matrix resulting from the multiplication, which implies that each thread will calculate a dot product between a row of matrix A and a column of matrix B

# In[ ]:


cudaKernel2 = '''
__global__ void matrixMul(float *A, float *B, float *C)
{
    int tid_x = blockDim.x * blockIdx.x + threadIdx.x; // Row
    int tid_y = blockDim.y * blockIdx.y + threadIdx.y; // Column
    int matrixDim = gridDim.x * blockDim.x;
    int tid   = matrixDim * tid_y + tid_x; // element i,j
    
    float  aux=0.0f;
    
    for ( int i=0 ; i<matrixDim ; i++ ){
        //          
        aux += A[matrixDim * tid_y + i]*B[matrixDim * i + tid_x] ;
    
    }
    
    C[tid] = aux;
             
}
'''


# In[ ]:


myCode = SourceModule(cudaKernel2)
mulMatrix = myCode.get_function("matrixMul")


# In[ ]:


mulMatrix(a_gpu,b_gpu,c_gpu,block=cuBlock,grid=cuGrid)


# In[ ]:


dotAB = np.dot(a_cpu,b_cpu)


# In[ ]:


dotAB


# In[ ]:


c_gpu


# In[ ]:


diff = np.abs(c_gpu.get()-dotAB)
np.sum(np.sum(diff))


# In[ ]:


plt.imshow(diff,interpolation='none')
plt.colorbar()


# In[ ]:


getKernelInfo(mulMatrix,nthreads)


# # Performance testing

# We will now check the performance of pyCUDA vs. Numpy

# For reference, if we use single precision, we use 4 bytes per array element, while in double precision we will have 8 bytes per element. Thus, in double precision the storage requirements for the different configurations are:

# |Points|size 1D (Mb)|size 2D (Mb)|size 3D (Mb)|
# |:----:|:----------:|:----------:|:----------:|
# |128|0.001|0.125|16|
# |256|0.002|0.5|128|
# |512|0.004|2|1024|
# |1024|0.008|8|8192|

# In[ ]:


from time import time
def myColorRand():
    return (np.random.random(),np.random.random(),np.random.random())


# In[ ]:


dimension = [2**i for i in range(5,25) ]
myPrec = presCPU


# In[ ]:


dimension 


# ### Vector addition

# In[ ]:


nLoops = 100
timeCPU = []
for n in dimension:
    v1_cpu = np.random.random(n).astype(myPrec)
    v2_cpu = np.random.random(n).astype(myPrec)
    tMean = 0
    for i in range(nLoops):
        t = time() 
        v = v1_cpu+v2_cpu
        t = time() - t
        tMean += t/nLoops
    timeCPU.append(tMean)


# In[ ]:


plt.figure(1,figsize=(10,6), dpi=200)
plt.loglog(dimension,timeCPU,'b-*')
plt.ylabel('Time (sec)')
plt.xlabel('N')
plt.xticks(dimension, dimension, rotation='vertical')


# **GPU version**

# Using the handy GPUArray class

# In[ ]:


timeGPU1 = []
bandWidth1 = []
for n in dimension:
    v1_cpu = np.random.random(n).astype(myPrec)
    v2_cpu = np.random.random(n).astype(myPrec)
    t1Mean = 0
    t2Mean = 0
    for i in range(nLoops):
        t = time()
        vaux = gpuarray.to_gpu(v1_cpu)
        t = time() -t
        t1Mean += t/nLoops
    bandWidth1.append(t1Mean)
    v1_gpu = gpuarray.to_gpu(v1_cpu) 
    v2_gpu = gpuarray.to_gpu(v2_cpu)
    for i in range(nLoops):
        t = time()
        v = v1_gpu+v2_gpu
        t = time() -t
        t2Mean += t/nLoops
    timeGPU1.append(t2Mean)
    v1_gpu.gpudata.free()
    v2_gpu.gpudata.free()
    v.gpudata.free()


# In[ ]:


plt.figure(1,figsize=(10,6))
plt.loglog(dimension,timeGPU1,'r-*',label='GPUArray')
plt.loglog(dimension,timeCPU,'b-*',label='CPU')
plt.ylabel('Time (sec)')
plt.xlabel('N')
plt.xticks(dimension, dimension, rotation='vertical')
plt.legend(loc=1,labelspacing=0.5,fancybox=True, handlelength=1.5, borderaxespad=0.25, borderpad=0.25)


# If you're running on an NVidia K40, you will see a jump in computation time from N=131,072 to N=262,144 for the GPU Simple method. This could be related to the different types of memory that are present in a GPU - and the GPUArray method switching from one type of memory to another, to accomodate the larger matrix size. 131,072 Floats (524,288 bytes) fit nicely in the shared memory of a K40 (which totals 737,280 bytes, divided over 15 SMs), while 262,144 floats would be too large and may simply be stored in global memory. Unfortunately, we cannot verify this theory, since the GPUArray method does not allow us explicit control over which memory is used.

# You may note that the time taken for N=32 is relatively long. This is because for the first iteration, some GPU initialization has to be performed. Below, we rerun the same code - you will see that since initialization is already done, the time taken for N=32 is now similar to that of N=64.

# In[ ]:


timeGPU1 = []
bandWidth1 = []
for n in dimension:
    v1_cpu = np.random.random(n).astype(myPrec)
    v2_cpu = np.random.random(n).astype(myPrec)
    t1Mean = 0
    t2Mean = 0
    for i in range(nLoops):
        t = time()
        vaux = gpuarray.to_gpu(v1_cpu)
        t = time() -t
        t1Mean += t/nLoops
    bandWidth1.append(t1Mean)
    v1_gpu = gpuarray.to_gpu(v1_cpu) 
    v2_gpu = gpuarray.to_gpu(v2_cpu)
    for i in range(nLoops):
        t = time()
        v = v1_gpu+v2_gpu
        t = time() -t
        t2Mean += t/nLoops
    timeGPU1.append(t2Mean)
    v1_gpu.gpudata.free()
    v2_gpu.gpudata.free()
    v.gpudata.free()


# In[ ]:


plt.figure(1,figsize=(10,6))
plt.loglog(dimension,timeGPU1,'r-*',label='GPUArray')
plt.loglog(dimension,timeCPU,'b-*',label='CPU')
plt.ylabel('Time (sec)')
plt.xlabel('N')
plt.xticks(dimension, dimension, rotation='vertical')
plt.legend(loc=1,labelspacing=0.5,fancybox=True, handlelength=1.5, borderaxespad=0.25, borderpad=0.25)


# In[ ]:


# plt.figure(1,figsize=(10,6))

a = np.array(timeGPU1)
b = np.array(timeCPU)
plt.semilogx(dimension,b/a,'r-*',label='CPUtime/GPUtime')
plt.ylabel('SpeedUp x')
plt.xlabel('N')
plt.title('SpeedUP')
plt.xticks(dimension, dimension, rotation='vertical')
plt.legend(loc=1,labelspacing=0.5,fancybox=True, handlelength=1.5, borderaxespad=0.25, borderpad=0.25)


# The speedup shows that the GPU is only faster than the CPU for sufficiently large matrices.

# Memory transfer cost

# In[ ]:


plt.figure(1,figsize=(10,6))
sizeMB = np.array(dimension)/(2.**20)
print (sizeMB)
plt.loglog(sizeMB,bandWidth1,'m-+',label='GPU copy  HostToDevice')
plt.loglog(sizeMB,timeGPU1,'r-*',label='GPUArray')
plt.ylabel('Time (sec)')
plt.xlabel('Memory (MB)')
plt.xticks(sizeMB, sizeMB, rotation='vertical')
plt.legend(loc=1,labelspacing=0.5,fancybox=True, handlelength=1.5, borderaxespad=0.25, borderpad=0.25)


# Implementation using the elementwise type functions

# In[ ]:


from pycuda.elementwise import ElementwiseKernel
myCudaFunc = ElementwiseKernel(arguments = "float *a, float *b, float *c",
                               operation = "c[i] = a[i]+b[i]",
                               name = "mySumK")


# In[ ]:


import pycuda.driver as drv
start = drv.Event()
end = drv.Event()


# Note that, to avoid timing the initialization, we do one 'warm-up' calculation before we start the actual loop over 'n'. This prevents the spike we initially observed for N=32 for GPU Simple method.

# In[ ]:


timeGPU2 = []
#Warmup
v1_cpu = np.random.random(n).astype(myPrec)
v2_cpu = np.random.random(n).astype(myPrec)
v1_gpu = gpuarray.to_gpu(v1_cpu) 
v2_gpu = gpuarray.to_gpu(v2_cpu)
vr_gpu  = gpuarray.to_gpu(v2_cpu)
myCudaFunc(v1_gpu,v2_gpu,vr_gpu)
v1_gpu.gpudata.free()
v2_gpu.gpudata.free()
vr_gpu.gpudata.free()
#End of warmup
for n in dimension:
    v1_cpu = np.random.random(n).astype(myPrec)
    v2_cpu = np.random.random(n).astype(myPrec)
    v1_gpu = gpuarray.to_gpu(v1_cpu) 
    v2_gpu = gpuarray.to_gpu(v2_cpu)
    vr_gpu  = gpuarray.to_gpu(v2_cpu)
    t3Mean=0
    for i in range(nLoops):
        start.record()
        myCudaFunc(v1_gpu,v2_gpu,vr_gpu)
        end.record()
        end.synchronize()
        secs = start.time_till(end)*1e-3
        t3Mean+=secs/nLoops
    timeGPU2.append(t3Mean)
    v1_gpu.gpudata.free()
    v2_gpu.gpudata.free()
    vr_gpu.gpudata.free()


# In[ ]:


plt.figure(1,figsize=(10,6))
plt.loglog(dimension,timeGPU1,'r-*',label='GPUArray')
plt.loglog(dimension,timeGPU2,'g-*',label='GPU Elementwise Sum')
plt.ylabel('Time (sec)')
plt.xlabel('N')
plt.xticks(dimension, dimension, rotation='vertical')
plt.legend(loc=1,labelspacing=0.5,fancybox=True, handlelength=1.5, borderaxespad=0.25, borderpad=0.25)


# Note that the GPU Elementwise Sum method does not contain the same jump that GPU Simple Sum has from N=131,072 to N=262,144. For both methods, we have no control over in which memory dat is stored on the GPU, or on how calculations are scheduled, but the GPU Elementwise implementation seems to make more clever use of the hardware than the GPUArrays.

# In[ ]:


plt.figure(1,figsize=(10,6))
plt.title('GPU: {0}, in {1} precision'.format(dev.name(),presGPU),size=22)
a=np.array(timeGPU1)
b=np.array(timeGPU2)
plt.semilogx(dimension,a/b,'r-*',label='GPUArray / GPU Elementwise')
plt.ylabel(' Speedup x')
plt.xlabel('N')
plt.xticks(dimension, dimension, rotation='vertical')
plt.legend(loc=1,labelspacing=0.5,fancybox=True, handlelength=1.5, borderaxespad=0.25, borderpad=0.25)


# In[ ]:


plt.figure(1,figsize=(10,6))
plt.title('GPU: {0}, in {1} precision'.format(dev.name(),presGPU),size=22)
a=np.array(timeCPU)
b=np.array(timeGPU2)
plt.semilogx(dimension,a/b,'r-*',label='CPU / GPU Elementwise')
plt.ylabel(' Speedup x')
plt.xlabel('N')
plt.xticks(dimension, dimension, rotation='vertical')
plt.legend(loc=1,labelspacing=0.5,fancybox=True, handlelength=1.5, borderaxespad=0.25, borderpad=0.25)


# Finally we have PyCUDA kernel-based implementation using SourceModule. In this implementation we can test the variation of the size of the block.

# In[ ]:


cudaCode = open("gpuAdd.cu","r")
cudaCode = cudaCode.read()
cudaCode = cudaCode.replace('float',presGPU )
print (cudaCode)
myCode = SourceModule(cudaCode)
vectorAddKernel = myCode.get_function("vectorAdd")
vectorAddKernel.prepare('PPP')


# In[ ]:


timeGPU3 = []
occupancyMesure=[]
for nt in [32,64,128,256,512,1024]:
    aux = []
    auxOcc = []
    for n in dimension:
        v1_cpu = np.random.random(n).astype(myPrec)
        v2_cpu = np.random.random(n).astype(myPrec)
        v1_gpu = gpuarray.to_gpu(v1_cpu) 
        v2_gpu = gpuarray.to_gpu(v2_cpu)
        vr_gpu  = gpuarray.to_gpu(v2_cpu)
        cudaBlock = (nt,1,1) 
        cudaGrid    = ((n+nt-1)/nt,1,1)
        
        cudaCode = open("gpuAdd.cu","r")
        cudaCode = cudaCode.read()
        cudaCode = cudaCode.replace('float',presGPU )
        downVar = ['blockDim.x','blockDim.y','blockDim.z','gridDim.x','gridDim.y','gridDim.z']
        upVar      = [str(cudaBlock[0]),str(cudaBlock[1]),str(cudaBlock[2]),
                     str(cudaGrid[0]),str(cudaGrid[1]),str(cudaGrid[2])]
        dicVarOptim = dict(zip(downVar,upVar))
        for i in downVar:
            cudaCode = cudaCode.replace(i,dicVarOptim[i])
        #print cudaCode
        myCode = SourceModule(cudaCode)
        vectorAddKernel = myCode.get_function("vectorAdd")
        vectorAddKernel.prepare('PPP')
        
        print ('\n Size={0}, threadsPerBlock={1}'.format(n,nt))
        print (cudaBlock,cudaGrid)
        t5Mean = 0
        #for i in range(nLoops):
            #timeAux = vectorAddKernel.prepared_timed_call(cudaGrid,cudaBlock,v1_gpu.gpudata,v2_gpu.gpudata,vr_gpu.gpudata)
            #t5Mean += timeAux()/nLoops
        auxOcc.append(getKernelInfo(vectorAddKernel,cudaBlock[0]*cudaBlock[1]*cudaBlock[2]))
        aux.append(t5Mean)
        v1_gpu.gpudata.free()
        v2_gpu.gpudata.free()
        vr_gpu.gpudata.free()
    timeGPU3.append(aux)
    occupancyMesure.append(auxOcc)


# In[ ]:


timeGPU3[0]


# In[ ]:


plt.figure(1,figsize=(10,6),dpi=100)
plt.semilogx(dimension,timeGPU1,'y-*',label='GPUArray')
plt.semilogx(dimension,timeGPU2,'g-*',label='GPU Elementwise Sum')
count = 0
for nt in [32,64,128,256,512,1024]:
    plt.loglog(dimension,timeGPU3[count],'-*',label='GPU Kernel, block={0}'.format(nt),color=(0,1./(count+1),1))
    count+=1
plt.ylabel('Time (sec)')
plt.xlabel('N')
plt.xticks(dimension, dimension, rotation='vertical')
plt.legend(loc=2,labelspacing=0.5,fancybox=True, handlelength=1.5, borderaxespad=0.25, borderpad=0.25)


# In[ ]:


plt.figure(1,figsize=(10,6),dpi=200)
count = 0
for nt in [32,64,128]:
    plt.semilogx(dimension,occupancyMesure[count],'-*',label='GPU Kernel, block={0}'.format(nt),color=(0,1./(2*count+1),0), alpha=0.5)
    count+=1
plt.ylabel('Occupancy')
plt.xlabel('N data')
plt.ylim(0,1.2)
plt.xticks(dimension, dimension, rotation='vertical')
plt.legend(loc=2,labelspacing=0.5,fancybox=True, handlelength=1.5, borderaxespad=0.25, borderpad=0.25)


# In[ ]:


plt.figure(1,figsize=(10,6),dpi=200)
count = 3
for nt in [256,512,1024]:
    plt.semilogx(dimension,occupancyMesure[count],'-*',label='GPU Kernel, block={0}'.format(nt),color=(0.5,1./(2*count+1),1./count), alpha=0.9)
    count+=1
plt.ylabel('Occupancy')
plt.xlabel('N data')
plt.ylim(0,1.2)
plt.xticks(dimension, dimension, rotation='vertical')
plt.legend(loc=2,labelspacing=0.5,fancybox=True, handlelength=1.5, borderaxespad=0.25, borderpad=0.25)


# In[ ]:


plt.figure(1,figsize=(12,8),dpi=300)
plt.title('GPU: {0}, in {1} presicion'.format(dev.name(),presGPU),size=22)
plt.loglog(dimension,timeGPU1,'y-*',label='GPUArray')
plt.loglog(dimension,timeGPU2,'g-*',label='GPU Elementwise Sum')
count = 0
for nt in [32,64,128,256,512,1024]:
    plt.loglog(dimension,timeGPU3[count],'-*',label='GPU Kernel, block={0}'.format(nt),color=myColorRand())
    count+=1
plt.ylabel('Time (seg)')
plt.xlabel('N')
plt.xticks(dimension, dimension, rotation='vertical')
plt.legend(loc=2,labelspacing=0.5,fancybox=True, handlelength=1.5, borderaxespad=0.25, borderpad=0.25)


# We see that all GPU kernels with block size >= 128 perform similar. Thus, it seems that for this specific kernel, an occupancy of ~0.67 was required to hide memory latency and fully use the available memory bandwidth.

# **NOTICE**:  It would be better to compute an average over the multiple runs

# In[ ]:


myColorRand()


# In[ ]:


plt.figure(1,figsize=(12,8),dpi=200)
plt.loglog(dimension,timeGPU1,'y-*',label='GPUArray', alpha=0.8,linewidth=3)
plt.loglog(dimension,timeGPU2,'g-*',label='GPU Elementwise Sum', alpha=0.8,linewidth=3)
count = 0
for nt in [32,64,128]:
    plt.loglog(dimension,timeGPU3[count],'-*',label='GPU Kernel, block={0}'.format(nt),color=myColorRand(), alpha=0.8,linewidth=3)
    count+=1
plt.ylabel('Time (seg)')
plt.xlabel('N')
plt.xticks(dimension, dimension, rotation='vertical')
plt.legend(loc=2,labelspacing=0.5,fancybox=True, handlelength=1.5, borderaxespad=0.25, borderpad=0.25)


# In[ ]:


np.random.random()


# In[ ]:


plt.figure(1,figsize=(12,8),dpi=200)
plt.title('GPU: {0}, in {1} presicion'.format(dev.name(),presGPU),size=22)
#plt.loglog(dimension,timeGPU1,'y-*',label='GPU Simple Sum')
plt.loglog(dimension,timeGPU2,'g-*',label='GPU Elementwise Sum',alpha=0.8,linewidth=3)
count = 3
for nt in [256,512,1024]:
    plt.loglog(dimension,timeGPU3[count],'-*',label='GPU Kernel, block={0}'.format(nt),color=myColorRand(), alpha=0.8,linewidth=3)
    count+=1
plt.ylabel('Time (seg)')
plt.xlabel('N')
plt.ylim(1e-5,2e-3)
plt.xticks(dimension, dimension, rotation='vertical')
plt.legend(loc=2,labelspacing=0.5,fancybox=True, handlelength=1.5, borderaxespad=0.25, borderpad=0.25)


# ### SUMMARY
# You have seen three Python methods to sum (/multiply) two matrices using a GPU:
# * Using GPUArrays
# * Using Elementwise kernels
# * Using CUDA kernels through the SourceModule function
# 
# In general, there is a tradeoff between simplicity and flexibility/control. GPUArrays methods are the easiest to use, but operations are limited to what is implemented for these GPUArrays. Elementwise kernels are slightly more flexible, allowing you to specify your own element-wise operation. The performance of the Elementwise kernels for the example of vector addition was also better, but there is no guarantee that will be the case for other operations as well.
# 
# Both GPUArrays and Elementwise kernels do not allow any control over which GPU memory is used and how calculations are scheduled on the GPU. Using CUDA kernels is by far the most flexible, as you can write a kernel for whatever operation you want, and you have full control over which memory is used, and how threads are scheduled. For the CUDA kernel method, it is the programmers responsibility to use the memory and thread scheduling efficiently, making it the most complex and (if used correctly) fastest solution of the three.

# In[ ]:




