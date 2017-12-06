from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

start = MPI.Wtime()

if rank == 0:
  data = {'a': 7, 'b': 3.14} 
  req = comm.isend(data, dest=1, tag=11)
  req.wait()
elif rank == 1:
  req = comm.irecv(source=0, tag=11)
  data = req.wait()

end = MPI.Wtime()
elapsed = end - start

print("Rank %d: Elapsed time is %f seconds.  Data is %r." % (rank, elapsed, data))

