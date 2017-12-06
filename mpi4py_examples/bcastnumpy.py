from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
  data = np.arange(100, dtype='i')
else:
  data = np.empty(100, dtype='i')

start = MPI.Wtime()
comm.Bcast(data, root=0)
end = MPI.Wtime()
elapsed = end - start

for i in range(100):
  assert data[i] == i

print('Rank %d: elapsed time is %f.' % (rank, elapsed))
