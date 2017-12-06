# https://www.datasciencecentral.com/profiles/blogs/python-implementing-a-k-means-algorithm-with-sklearn
# By Michael Grogan

import pandas
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import time
 
variables = pandas.read_csv('sample_stocks.csv')

Y = variables[['returns']]

X = variables[['dividendyield']]

start = time.time()

Nc = range(1, 40)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans

score = [kmeans[i].fit(Y).score(Y) for i in range(len(kmeans))]
score
stop = time.time()

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
