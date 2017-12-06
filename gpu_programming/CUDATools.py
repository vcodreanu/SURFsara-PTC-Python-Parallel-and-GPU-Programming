from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import pycuda.driver as cuda
from pycuda.tools import DeviceData
from pycuda.tools import OccupancyRecord as occupancy

# Difference between int and long long for id's inside kernels take place at 2D arrays of
# length above 46341 and for 3D above 1290 point per dimension, this appears for array size of 17 GBs

def setDevice(ndev = None):
      ''' To use CUDA or OpenCL you need a context and a device to stablish the context o                   communication '''
      import pycuda.autoinit
      nDevices = cuda.Device.count()
      print "Available Devices:"
      for i in range(nDevices):
            dev = cuda.Device( i )
            print "  Device {0}: {1}".format( i, dev.name() )
      devNumber = 0
      if nDevices > 1:
            if ndev == None:
                devNumber = int(raw_input("Select device number: "))
            else:
                devNumber = ndev
      dev = cuda.Device( devNumber)
      cuda.Context.pop()  #Disable previus CUDA context
      ctxCUDA = dev.make_context()
      print "Using device {0}: {1}".format( devNumber, dev.name() )
      return ctxCUDA, dev


def getKernelInfo(kernel,nthreads, rt=True):
    ''' This function returns info about kernels theoretical performance, but warning is not trivial to optimize! '''
    shared=kernel.shared_size_bytes
    regs=kernel.num_regs
    local=kernel.local_size_bytes
    const=kernel.const_size_bytes
    mbpt=kernel.max_threads_per_block
    #threads =  #self.block_size_x* self.block_size_y* self.block_size_z
    occupy = occupancy(devdata, nthreads, shared_mem=shared, registers=regs)
    print "==Kernel Memory=="
    print("""Local:        {0}
Shared:       {1}
Registers:    {2}
Const:        {3}
Max Threads/B:{4}""".format(local,shared,regs,const,mbpt))
    print "==Occupancy=="
    print("""Blocks executed by MP: {0}
Limited by:            {1}
Warps executed by MP:  {2}
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
    print "Call the function {0} times takes in GPU {1} seconds.\n".format(ntimes,timeGPU)
    print "{0} seconds per call".format(timeGPU/ntimes)
    return timeGPU

def precisionCU(p = 'float'):
  '''(presicion) p = float,cfloat,double,cdouble'''
  if p == 'float':
    return np.float32, p, p
  if p == 'cfloat':
    return np.complex64, 'pycuda::complex<float>', p
  if p == 'double':
    return np.float64, p, p
  if p == 'cdouble':
    return np.complex128, 'pycuda::complex<double>', p

def optKernels(kFile,pres,subBlGr = False, cuB=(1,1,1), cuG=(1,1,1),compiling=False,myDir=None):

    if pres == 'float':
      cString = 'f'
      kFile = kFile.replace('cuPres', pres)
      kFile = kFile.replace('cuStr', cString)
      kFile = kFile.replace('cuName', pres) # for textures
    if pres == 'double':
      cString = ''
      kFile = kFile.replace('cuPres', pres)
      kFile = kFile.replace('cuStr', cString)
      kFile = kFile.replace('cuName', pres)
    if pres == 'cfloat':
      cString = ''
      presicion = 'pycuda::complex<float>'
      kFile = kFile.replace('cuPres', presicion)
      kFile = kFile.replace('cuStr', cString)
      kFile = kFile.replace('cuName', pres)
    if pres == 'cdouble':
      cString = ''
      presicion = 'pycuda::complex<double>'
      kFile = kFile.replace('cuPres', presicion)
      kFile = kFile.replace('cuStr', cString)
      kFile = kFile.replace('cuName', pres)

    if subBlGr:
        downVar = ['blockDim.x','blockDim.y','blockDim.z','gridDim.x','gridDim.y','gridDim.z']
        upVar      = [str(cuB[0]),str(cuB[1]),str(cuB[2]),
                      str(cuG[0]),str(cuG[1]),str(cuG[2])]
        dicVarOptim = dict(zip(downVar,upVar))
        for i in downVar:
            kFile = kFile.replace(i,dicVarOptim[i])
    if compiling:
      kFile = SourceModule(kFile,include_dirs=[myDir])
    return kFile

def getFreeMemory(show=True):
    ''' Return the free memory of the device,. Very usful to look for save device memory '''
    Mb = 1024.*1024.
    Mbytes = float(cuda.mem_get_info()[0])/Mb
    if show:
      print "Free Global Memory: %f Mbytes" %Mbytes

    return cuda.mem_get_info()[0]/Mb

ctx,device = setDevice()
devdata = DeviceData(device)
