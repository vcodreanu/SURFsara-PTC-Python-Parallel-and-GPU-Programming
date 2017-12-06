from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# initialize data
if rank == 0:
  data = numpy.arange(1000, dtype='i')
elif rank == 1:
  data = numpy.empty(1000, dtype='i')

# measure communication time
start = MPI.Wtime()
if rank == 0:
  comm.Send([data, MPI.INT], dest=1, tag=77)
elif rank == 1:
  comm.Recv([data, MPI.INT], source=0, tag=77)
end = MPI.Wtime()
elapsed = end - start

print("Rank %d: Elapsed time is %f seconds." % (rank, elapsed))

