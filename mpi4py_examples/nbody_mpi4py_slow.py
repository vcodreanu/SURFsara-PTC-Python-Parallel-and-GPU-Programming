from mpi4py import MPI
from types import FunctionType
import numpy as np

class Pool(object):
  """Process pool using MPI."""
  
  def __init__(self):
    self.f = None
    self.size = size
    self.rank = rank

  def wait(self):
    if self.rank ==0:
      raise RuntimeError("Proc 0 cannot wait!")
    status = MPI.Status()
    while True:
      task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
      if not task:
        break
      if isinstance(task, FunctionType):
        self.f = task
        continue
      result = self.f(task)
      comm.isend(result, dest=0, tag=status.tag)

  def map(self, f, tasks):
    N = len(tasks)
    numprocs = self.size
    nump_minus_1 = numprocs - 1
    if self.rank != 0:
      self.wait()
      return

    if f is not self.f:
      self.f = f
      requests = []
      for proc in range(1, numprocs):
        req = comm.isend(f, dest=proc)
        requests.append(req)
      MPI.Request.waitall(requests)

    requests = []
    for i, task in enumerate(tasks):
      req = comm.isend(task, dest=(i%nump_minus_1)+1, tag=i)
      requests.append(req)
    MPI.Request.waitall(requests)

    results = []
    for i in range(N):
      result = comm.recv(source=(i%nump_minus_1)+1, tag=i)
      results.append(result)
    return results      

  def __del__(self):
    if self.rank == 0:
      for proc in range(1, self.size):
        comm.isend(False, dest=proc)

def timestep_i(args):
  """Computes the next position and velocity for the ith mass."""
  i, position, velocity, G, mass, dt = args
  pos_i = acceleration(i, position, G, mass)
  new_velocity_i = pos_i * dt + velocity[i]
  new_pos_i = pos_i * dt**2 + velocity[i] * dt +position[i]
  return i, new_pos_i, new_velocity_i
 
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


def initial_cond(N, Dim):
  """Generates initial conditions for N unity masses at rest
  starting at random positions in D-dimensional space.
  """
  position0 = np.random.rand(N, Dim)
  velocity0 = np.zeros((N, Dim), dtype=float)
  mass = np.ones(N, dtype=float)
  return position0, velocity0, mass 


def timestep(position, velocity, G, mass, dt, pool):
  """Computes the next position and velocity for all masses given
  initial conditions and a time step size.
  """
  N = len(position)
  tasks = [(i, position, velocity, G, mass, dt) for i in range(N)]
  results = pool.map(timestep_i, tasks)
  new_pos = np.empty(position.shape, dtype=float)
  new_velocity = np.empty(velocity.shape, dtype=float)
  for i, new_pos_i, new_velocity_i in results:
    new_pos[i] = new_pos_i
    new_velocity[i] = new_velocity_i
  return new_pos, new_velocity


def simulate(timesteps, G, dt, position0, velocity0, mass):
  """N-body simulation of certain timesteps."""
  position, velocity = position0, velocity0
  pool = Pool()
    
  if rank == 0:
    for step in range(timesteps):
      new_pos, new_velocity = timestep(position, velocity, G, mass, dt, pool)
      position , velocity = new_pos, new_velocity
  else:
    pool.wait()  


if __name__ == "__main__":
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()

  N = 256
  Dim = 3
  position0, velocity0, mass = initial_cond(N, Dim)
  if rank == 0:
    start = MPI.Wtime()
  simulate(600, 1.0, 1e-3, position0, velocity0, mass)
  if rank == 0:
    stop = MPI.Wtime()
    elapsed = stop - start
    print('Number of processes: %d, runtime is %f seconds' % (size, elapsed))
