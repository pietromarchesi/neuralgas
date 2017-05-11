from __future__ import division

import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

from neuralgas.oss_gwr import gwr
from neuralgas.oss_gwr_h import gwr_h_unimodal
from neuralgas.oss_gwr_h import _propagate_trajectories

iris = sklearn.datasets.load_iris()
X = iris.data
X1 = X[:,:2]
X2 = X[:,2:]
y = iris.target

g1 = gwr_h_unimodal(n_layers = 2, window_size=[1, 2],
                   gwr_pars=[{'act_thr':0.7},{'act_thr':0.7}])

g2 = gwr_h_unimodal(n_layers = 2, window_size=[1, 2],
                   gwr_pars=[{'act_thr':0.7},{'act_thr':0.7}])

g3 = gwr()

XX1 = g1.train(X1)
XX2 = g2.train(X2)
X3 = np.hstack((XX1,XX2))
g3.train(X3)
Xf = _propagate_trajectories(X3, network= g3, ws = 2)

# then you can take a oss_gwr

