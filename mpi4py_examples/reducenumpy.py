from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sendbuf = np.zeros(10, dtype='i') + rank
recvbuf = None
if rank == 0:
  recvbuf = np.zeros(10, dtype='i')
comm.Reduce([sendbuf, MPI.INT], [recvbuf, MPI.INT], op=MPI.SUM, root=0)

if rank == 0:
  sum = sum(range(size))
  assert (recvbuf[:]==sum).all()
  print(recvbuf)
