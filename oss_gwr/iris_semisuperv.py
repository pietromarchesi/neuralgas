from __future__ import division

import logging
import copy
import numpy as np
import sklearn.datasets
import sklearn.metrics
import matplotlib.pyplot as plt

from oss_gwr.oss import oss_gwr

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

iris = sklearn.datasets.load_iris()
X = iris.data
X.flags.writeable = False
y = iris.target
N = 50
ran = range(0,110,10)
acc = np.zeros([N,len(ran)])

for j in range(N):
    for k in range(len(ran)):

        # permute the data
        perm = np.random.permutation(X.shape[0])
        XX = X[perm, :]
        yy = y[perm]
        XX.setflags(write=False)

        # draw test and train data
        ind = np.random.choice(XX.shape[0], size=130, replace=False)
        mask = np.zeros_like(y, dtype=bool)
        mask[ind] = True
        X_train = X[mask, :]
        y_train = y[mask]
        X_test = X[~mask, :]
        y_test = y[~mask]

        # obfuscate the data
        i = ran[k]
        s = i * X_train.shape[0] // 100
        ind2 = np.random.choice(ind, size=s, replace=False)
        mask = np.zeros_like(y, dtype = bool)
        mask[ind2] = True
        yy[~mask] = -1
        yy.setflags(write=False)

        # classify
        gwr = oss_gwr(act_thr=0.35,max_age=500, kappa=1.05)
        gwr.train(XX, yy, n_epochs=20)
        y_pred = gwr.predict(X)
        a = sklearn.metrics.accuracy_score(y,y_pred)
        acc[j,k] = a

Acc = acc.mean(axis = 0)
std = acc.std(axis = 0)
f,ax = plt.subplots(1,1)
ax.set_xlabel('Labelled samples (%)')
ax.set_ylabel('Classification accuracy')
ax.errorbar(ran,Acc,yerr = std)
ax.plot(ran,np.linspace(0,1,len(ran)))
ax.set_xticks(ran)
ax.set_xticklabels(ran)
ax.set_ylim([0,1])

