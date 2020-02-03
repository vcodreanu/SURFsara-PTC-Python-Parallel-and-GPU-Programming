#!/usr/bin/env python
from mpi4py import MPI
import numpy as np
import time

import pycuda.autoinit
from pycuda import gpuarray

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()
sizes = 16000

print ("MPI comm_size {}".format(comm_size)) 

#define a float16 mpi datatype
mpi_float16 = MPI.BYTE.Create_contiguous(2).Commit()
MPI._typedict['e'] = mpi_float16
def sum_f16_cb(buffer_a, buffer_b, t):
    assert t == mpi_float16
    array_a = np.frombuffer(buffer_a, dtype='float16')
    array_b = np.frombuffer(buffer_b, dtype='float16')
    array_b += array_a
#create new OP
mpi_sum_f16 = MPI.Op.Create(sum_f16_cb, commute=True)

data = np.array([comm_rank] * sizes,dtype=np.float32)
data_gpu = gpuarray.to_gpu(data)
data_buf = data_gpu.gpudata.as_buffer(data_gpu.nbytes)

result = np.empty_like(data)
result_gpu = gpuarray.empty(data.shape, np.float32)
result_buf = result_gpu.gpudata.as_buffer(result_gpu.nbytes)

t1 = time.time()
comm.Allreduce([data,MPI.FLOAT], [result,MPI.FLOAT], op=MPI.SUM) #mpi_sum_f16)
#result_buf = comm.allreduce([data_buf,MPI.FLOAT],op=MPI.SUM) #mpi_sum_f16)
t1 = time.time() - t1

final_data = np.array([data] * sizes,dtype=np.float32)
final_data_gpu = gpuarray.to_gpu(final_data)
final_data_buf = final_data_gpu.gpudata.as_buffer(final_data_gpu.nbytes)

final_result = np.empty_like(final_data)
final_result_gpu = gpuarray.empty(final_data.shape, np.float32)
final_result_buf = final_result_gpu.gpudata.as_buffer(final_result_gpu.nbytes)

t2 = time.time()
#comm.Allreduce([final_data_buf,MPI.FLOAT], [final_result_buf,MPI.FLOAT], op=MPI.SUM) #mpi_sum_f16)
#final_result_buf = comm.allreduce([final_data_buf,MPI.SUM],op=MPI.SUM) #mpi_sum_f16)
t2 = time.time() - t2

#result_gpu.get(result, result_gpu.gpudata)
#print(result)
#final_result_gpu.get(final_result, final_result_gpu.gpudata)
#print(final_result)

if comm_rank == 0:
    print ("Elapsed time {}".format(t1+t2))
