from __future__ import division

import logging
import copy
import numpy as np
import networkx as nx
import sklearn.datasets
import sklearn.metrics
import matplotlib.pyplot as plt

from oss_gwr.oss import oss_gwr_supervised

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

iris = sklearn.datasets.load_iris()
X = iris.data
X.flags.writeable = False
y = iris.target
N = 100
ran = range(0,110,10)
acc = np.zeros([N,len(ran)])

graph = np.zeros([2, N,len(ran)])

for j in range(N):
    for k in range(len(ran)):

        # permute the data
        perm = np.random.permutation(X.shape[0])
        XX = X
        yy = y
        XX.setflags(write=False)

        # draw test and train data
        ind = np.random.choice(XX.shape[0], size=100, replace=False)
        mask = np.zeros_like(y, dtype=bool)
        mask[ind] = True
        X_train = XX[mask, :]
        y_train = yy[mask]
        X_test  = XX[~mask, :]
        y_test  = yy[~mask]

        # obfuscate the data
        i = ran[k]
        s = i * X_train.shape[0] // 100
        ind2 = np.random.choice(X_train.shape[0], size=s, replace=False)
        mask = np.zeros_like(y_train, dtype = bool)
        mask[ind2] = True
        y_train[~mask] = -1
        y_train.setflags(write=False)

        # classify
        gwr = oss_gwr_supervised(act_thr=0.75, max_age=500, kappa=1.05)
        gwr.train(X_train, y_train, n_epochs=20)
        y_pred = gwr.predict(X_test)
        a = sklearn.metrics.accuracy_score(y_test,y_pred)
        acc[j,k] = a

        Gcc = sorted(nx.connected_component_subgraphs(gwr.G), key=len,
                     reverse=True)
        G0 = len(Gcc[0].nodes())
        C = nx.average_clustering(gwr.G)
        graph[0,j,k] = G0
        graph[1,j,k] = C

Acc = acc.mean(axis = 0)
std = acc.std(axis = 0)

giant = graph[0].mean(axis = 0)
g_std = graph[0].std(axis = 0)

clust = graph[1].mean(axis = 0)
c_std = graph[1].std(axis = 0)

f,ax = plt.subplots(1,1)
ax.set_xlabel('Labelled samples (%)')
ax.set_ylabel('Classification accuracy')
ax.errorbar(ran,Acc,yerr = std)
ax.plot(ran,np.linspace(0,1,len(ran)))
ax.set_xticks(ran)
ax.set_xticklabels(ran)
ax.set_ylim([0,1])

f2, (ax1, ax2) = plt.subplots(1,2)
ax1.errorbar(ran,giant,yerr=g_std)
ax2.errorbar(ran,clust,yerr=c_std)
ax1.set_xticks(ran)
ax1.set_xticklabels(ran)
ax2.set_xticks(ran)
ax2.set_xticklabels(ran)
