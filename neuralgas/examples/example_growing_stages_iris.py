from __future__ import division

import sklearn.datasets
import matplotlib.pyplot as plt

from neuralgas.oss_gwr import gwr

iris = sklearn.datasets.load_iris()
X = iris.data
y = iris.target

g1 = gwr(act_thr=0.75, random_state=12345)
g1.train(X[:,:2], n_epochs=1)
Xg1 = g1.get_positions()
g1.train(X[:,:2], n_epochs=19, warm_start=True)
Xg1_ = g1.get_positions()


g2 = gwr(act_thr=0.75, random_state=1234)
g2.train(X[:,2:], n_epochs=1)
Xg2 = g2.get_positions()
g2.train(X[:,2:], n_epochs=19, warm_start=True)
Xg2_ = g2.get_positions()

marks=15
f, ax = plt.subplots(2,3)
ax[0,0].scatter(X[:,0],X[:,1], s=marks)
ymin, ymax = ax[0,0].get_ylim()
xmin, xmax = ax[0,0].get_xlim()
ax[0,1].scatter(Xg1[:,0], Xg1[:,1], s=marks)
ax[0,2].scatter(Xg1_[:,0], Xg1_[:,1],s=marks)
for i in range(1,3):
    ax[0,i].set_xlim([xmin,xmax])
    ax[0,i].set_ylim([ymin,ymax])
ax[0,0].set_title('Original data')
ax[0,1].set_title('Gas - Epoch 1')
ax[0,2].set_title('Gas - Epoch 20')


ax[1,0].scatter(X[:,2],X[:,3], s=marks)
ymin, ymax = ax[1,0].get_ylim()
xmin, xmax = ax[1,0].get_xlim()
ax[1,1].scatter(Xg2[:,0], Xg2[:,1], s=marks)
ax[1,2].scatter(Xg2_[:,0], Xg2_[:,1],s=marks)
for i in range(1,3):
    ax[1,i].set_xlim([xmin,xmax])
    ax[1,i].set_ylim([ymin,ymax])
