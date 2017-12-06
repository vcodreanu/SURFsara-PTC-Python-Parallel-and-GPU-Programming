from mpi4py import MPI
import numpy as np

def remove_i(arr, i):
  """Drops the ith element of an array."""
  shape = (arr.shape[0]-1,) + arr.shape[1:]
  new_arr = np.empty(shape, dtype=float)
  new_arr[:i] = arr[:i]
  new_arr[i:] = arr[i+1:]
  return new_arr


def acceleration(i, position, G, mass):
  """The acceleration of the ith mass."""
  ith_pos = position[i]
  rest_pos = remove_i(position, i)
  rest_mass = remove_i(mass, i)
  diff = rest_pos - ith_pos
  mag3 = np.sum(diff**2, axis=1)**1.5
  accel = G * np.sum(diff * (rest_mass / mag3)[:,np.newaxis], axis=0)
  return accel

def timestep_i(i, position, velocity, G, mass, dt):
  """Computes the next position and velocity for the ith mass"""
  accel_i = acceleration(i, position, G, mass)
  new_velocity_i= accel_i * dt + velocity[i]
  new_pos_i = accel_i * dt ** 2 + velocity[i] * dt + position[i]
  return new_pos_i, new_velocity_i

def timestep(position, velocity, G, mass, dt):
  """Computes the next position and velocity for all masses given
  initial conditions and a time step size.
  Divide N-body in to size trunks. Each process calculate
  the new position and velocity of N_local-body.
  """
  N_local = N // size
  new_pos_local = np.empty([N_local, position.shape[1]], dtype=float)
  new_velocity_local = np.empty([N_local, velocity.shape[1]], dtype=float)
  for i in range(N_local):
    new_pos_local[i], new_velocity_local[i] = timestep_i(rank*N_local+i, position, velocity, G, mass, dt)
  return new_pos_local, new_velocity_local


def initial_cond(N, Dim):
  """Generates initial conditions for N unity masses at rest
  starting at random positions in D-dimensional space.
  """
  velocity0 = np.zeros([N, Dim], dtype=float)
  mass = np.ones(N, dtype=float)
  if rank == 0:
    position0 = np.random.rand(N, Dim)
  else:
    position0 = np.empty([N, Dim], dtype=float)
  comm.Bcast(position0, root=0)
  return position0, velocity0, mass 

def simulate(timesteps, G, dt, position0, velocity0, mass):
  """N-body simulation of certain timesteps."""
  position, velocity = position0, velocity0
  new_pos = np.empty(position.shape, dtype=float)
  new_velocity = np.empty(velocity.shape, dtype=float)
  for step in range(timesteps):
    new_pos_local, new_velocity_local = timestep(position, velocity, G, mass, dt)
    comm.Allgather([new_pos_local, MPI.FLOAT], [new_pos, MPI.FLOAT])
    comm.Allgather([new_velocity_local, MPI.FLOAT], [new_velocity, MPI.FLOAT])
    position , velocity = new_pos, new_velocity
  return position, velocity


if __name__ == "__main__":
  import h5py
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()
  N = 256
  if N%size != 0:
    print('Please provide a number that can be divided by %d' % N)
    raise Exception('Number of processes %d is not divisible by %d.' % (size, N))


  # Initialize N-body conditions
  # Set gravitational constant to 1
  Dim = 3
  G=1.0
  dt = 1.0e-3
  timesteps = 600
  path = '/Users/zhengm/src/play/python/mpi4py/'
  
  name = path + 'data_' + str(N).zfill(4) + 'nbody.h5'
  if rank == 0:
    start = MPI.Wtime()
  position0, velocity0, mass = initial_cond(N, Dim)
  
  position, velocity = simulate(timesteps, G, dt, position0, velocity0, mass)
  if rank ==0:
    stop = MPI.Wtime()
    elapsed = stop - start
    print('Elapsed time is: %f seconds' % elapsed)  

  with h5py.File(name, 'w') as hf:
    hf.create_dataset('position', data=position)
    hf.create_dataset('velocity', data=velocity)
