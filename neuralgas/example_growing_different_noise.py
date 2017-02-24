from __future__ import division

import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

from neuralgas.oss_gwr import gwr

iris = sklearn.datasets.load_iris()
X1 = np.random.normal(loc = 2, scale = 1.0, size=[1000,2])
X2 = np.random.normal(loc = 2, scale = 2, size=[1000,2])

g1 = gwr(act_thr=0.75, random_state=None, max_size=30)
g1.train(X1, n_epochs=30)
Xg1 = g1.get_positions()

g2 = gwr(act_thr=0.75, random_state=None, max_size=30)
g2.train(X2, n_epochs=30)
Xg2 = g2.get_positions()

marks=8
f, ax = plt.subplots(2,2)
ax[0,0].scatter(X1[:,0],X1[:,1], s=marks)
ax[1,0].scatter(X2[:,0],X2[:,1], s=marks)
ymin, ymax = ax[1,0].get_ylim()
xmin, xmax = ax[1,0].get_xlim()
ax[0,1].scatter(Xg1[:,0], Xg1[:,1], s=marks)

ax[0,0].set_ylim([ymin,ymax])
ax[0,0].set_xlim([xmin,xmax])
ax[0,1].set_ylim([ymin,ymax])
ax[0,1].set_xlim([xmin,xmax])
ax[0,0].set_title('Original data')
ax[0,1].set_title('Gas - Epoch 20')

ax[1,1].scatter(Xg2[:,0], Xg2[:,1], s=marks)
ax[1,1].set_ylim([ymin,ymax])
ax[1,1].set_xlim([xmin,xmax])


