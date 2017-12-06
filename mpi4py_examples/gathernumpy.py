from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sendbuf = np.zeros(100, dtype='i') + rank
recvbuf = None
if rank == 0:
  recvbuf = np.empty([size, 100], dtype='i')
comm.Gather(sendbuf, recvbuf, root=0)

if rank == 0:
  for i in range(size):
    assert np.allclose(recvbuf[i,:], i)
  print(recvbuf)
