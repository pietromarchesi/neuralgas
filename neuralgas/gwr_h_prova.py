from __future__ import division

import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

from neuralgas.oss_gwr_h import gwr_h_unimodal

iris = sklearn.datasets.load_iris()
X = iris.data
y = iris.target

g = gwr_h_unimodal()
g.H[0].act_thr = 0.9
Xf = g.train(X[:,2:], n_epochs=20)

g._get_activation_trajectories(X)