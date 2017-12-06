# https://www.datasciencecentral.com/profiles/blogs/python-implementing-a-k-means-algorithm-with-sklearn
# By Michael Grogan

import pandas
import numpy as np
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpi4py import MPI
 
variables = pandas.read_csv('sample_stocks.csv')

Y = variables[['returns']]

X = variables[['dividendyield']]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
  start = MPI.Wtime()

Nc = np.range(40)
Nb = np.array_split(Nc, size)

kmeans = [KMeans(n_clusters=i+1) for i in Nb[rank]]
myscore = [kmeans[i].fit(Y).score(Y) for i in range(len(kmeans))]
score_l = np.asarray(myscore)

score = None
if rank == 0:
  score = np.empty(40, dtype='d')

comm.Gather(score_l, score, root = 0)

if rank == 0:
  stop = MPI.Wtime()

  elapsed = stop - start

  print('Elapsed time: %f seconds' % elapsed)

  pl.plot(Nc,score)
  
  pl.xlabel('Number of Clusters')
  
  pl.ylabel('Score')
  
  pl.title('Elbow Curve')
  
  pl.show()

#pca = PCA(n_components=1).fit(Y)
#
#pca_d = pca.transform(Y)
#
#pca_c = pca.transform(X)
#
#kmeans=KMeans(n_clusters=3)
#
#kmeansoutput=kmeans.fit(Y)
#
#kmeansoutput
#
#
#pl.figure('3 Cluster K-Means')
#
#pl.scatter(pca_c[:, 0], pca_d[:, 0], c=kmeansoutput.labels_)
#
#pl.xlabel('Dividend Yield')
#
#pl.ylabel('Returns')
#
#pl.title('3 Cluster K-Means')
#
#pl.show()
