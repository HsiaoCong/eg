#encoding:utf-8

import nltk
from numpy import array

datas=[array(v) for v in [(1,0),(0,1),(1,1),(5,5),(5,4),(4,5)]]

#Kmeans聚类
km=nltk.cluster.kmeans.KMeansClusterer(num_means=2,distance=nltk.cluster.util.euclidean_distance)
km.cluster(datas)
for data in datas:
    print (str(data)+str(km.classify(data)))


#GAAC聚类
ga=nltk.cluster.gaac.GAAClusterer(num_clusters=3,normalise=True)
ga.cluster(vectors=datas)
ga.dendrogram().show()
for data in datas:
    print (str(data)+str(ga.classify(data)))


#混合混合聚类
emc=nltk.cluster.em.EMClusterer(initial_means=[[4,2],[4,2.01]])
emc.cluster(vectors=datas)
for data in datas:
    print (str(data)+str(emc.classify(data)))