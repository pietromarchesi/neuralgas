from __future__ import division

import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

from neuralgas.oss_gwr_h import gwr_h_unimodal
from neuralgas.oss_gwr_h import _propagate_trajectories

iris = sklearn.datasets.load_iris()
X = iris.data
X.flags.writeable = False
y = iris.target

g = gwr_h_unimodal(n_layers = 2, window_size=[1, 2],
                   gwr_pars=[{'act_thr':12233},{'eps_b':123456789}])

g.gwr_pars
#XX = g.train(X)
#XXX = g._get_activation_trajectories(X)

#np.testing.assert_array_equal(XX,XXX)




