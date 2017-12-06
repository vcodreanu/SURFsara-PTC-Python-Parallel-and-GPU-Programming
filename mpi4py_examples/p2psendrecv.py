from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# initialize data
if rank == 0:
  data = {'a': 7, 'b': 3.14}
# measure communication time
start = MPI.Wtime()
if rank == 0:
  comm.send(data, dest=1, tag=11)
elif rank == 1:
  data = comm.recv(source=0, tag=11)

end = MPI.Wtime()
elapsed = end - start

print("Rank %d: Elapsed time is %f seconds." % (rank, elapsed))

