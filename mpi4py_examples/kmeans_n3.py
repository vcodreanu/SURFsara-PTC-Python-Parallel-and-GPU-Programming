# https://www.datasciencecentral.com/profiles/blogs/python-implementing-a-k-means-algorithm-with-sklearn
# By Michael Grogan

import pandas
import numpy as np
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
 
variables = pandas.read_csv('sample_stocks.csv')

Y = variables[['returns']]

X = variables[['dividendyield']]

pca = PCA(n_components=1).fit(Y)

pca_d = pca.transform(Y)

pca_c = pca.transform(X)

kmeans=KMeans(n_clusters=3)

kmeansoutput=kmeans.fit(Y)

kmeansoutput


pl.figure('3 Cluster K-Means')

pl.scatter(pca_c[:, 0], pca_d[:, 0], c=kmeansoutput.labels_)

pl.xlabel('Dividend Yield')

pl.ylabel('Returns')

pl.title('3 Cluster K-Means')

pl.show()
