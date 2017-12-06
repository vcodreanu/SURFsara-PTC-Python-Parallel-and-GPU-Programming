from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print('Number of processes is %d.' %size)
print('Hello, I am process %d.' % rank)
