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
  result = G * np.sum(diff * (rest_mass / mag3)[:,np.newaxis], axis=0)
  return result


def timestep(position, velocity, G, mass, dt):
  """Computes the next position and velocity for all masses given
  initial conditions and a time step size.
  """
  N = len(position)
  new_pos = np.empty(position.shape, dtype=float)
  new_velocity = np.empty(velocity.shape, dtype=float)
  for i in range(N):
    acceleration_i = acceleration(i, position, G, mass)
    new_velocity[i] = acceleration_i * dt + velocity[i]
    new_pos[i] = acceleration_i * dt ** 2 + velocity[i] * dt + position[i]
  return new_pos, new_velocity


def initial_cond(N, Dim):
  """Generates initial conditions for N unity masses at rest
  starting at random positions in D-dimensional space.
  """
  position0 = np.random.rand(N, Dim)
  velocity0 = np.zeros((N, Dim), dtype=float)
  mass = np.ones(N, dtype=float)
  return position0, velocity0, mass 

def simulate(timesteps, G, dt, position0, velocity0, mass):
  """N-body simulation of certain timesteps."""
  position, velocity = position0, velocity0
  for step in range(timesteps):
    new_pos, new_velocity = timestep(position, velocity, G, mass, dt)
    position , velocity = new_pos, new_velocity
  return position, velocity


if __name__ == "__main__":
  import time
  import h5py

  N = 256
  # Initialize N-body conditions
  # Set gravitational constant to 1
  Dim = 3
  G=1.0
  dt = 1.0e-3
  timesteps = 600
  path = '/Users/zhengm/src/play/python/mpi4py/'
  
  name = path + 'data_' + str(N).zfill(4) + 'nbody_seq.h5'
  print(name)

  start = time.time()
  position0, velocity0, mass = initial_cond(N, Dim)
  position, velocity = simulate(timesteps, G, dt, position0, velocity0, mass)
  stop = time.time()
  elapsed = stop - start
  print('Elapsed time is: %f seconds' % elapsed)  

  with h5py.File(name, 'w') as hf:
    hf.create_dataset('position', data=position)
    hf.create_dataset('velocity', data=velocity)

