from __future__ import division

import logging
import numpy as np
import networkx as nx
import collections
from oss_gwr.oss import oss_gwr


class oss_h_unimodal():

    def __init__(self, layers = 2, window_size = [1,3], lab_ratio = 1/2,
                 network_pars = []):

        # TODO add network pars to the right networks
        self.layers      = layers
        self.window_size = window_size
        self.lab_ratio   = lab_ratio

        self.standard_pars = {'act_thr' : 0.35,
                              'fir_thr' : 0.1,
                              'eps_b'   : 0.1,
                              'eps_n'   : 0.01,
                              'tau_b'   : 0.3,
                              'tau_n'   : 0.1,
                              'kappa'   : 1.05,
                              'lab_thr' : 0.5,
                              'max_age' : 100}

        self.H = [oss_gwr() for _ in range(self.layers)]
        for i in range(self.layers):
            self.H[i].G = nx.Graph()

    def _get_trajectories_from_previous(self, X, layer, y = None):

        if hasattr(self.lab_ratio, '__len__'):
            lab_ratio = self.lab_ratio[layer]
        else:
            lab_ratio = self.lab_ratio
        ws = self.window_size[layer]
        Xt = np.zeros([X.shape[0]-(ws-1), X.shape[1]*ws])
        yt = np.zeros(X.shape[0]-(ws-1))
        for i in range((ws-1), X.shape[0]):

            x = np.zeros([0,])
            e = np.zeros([0,])
            for j in range(i+1-ws,i+1):
                if layer == 0:
                    p = X[j,0]
                    e = np.hstack((e, y[j]))
                else:
                    b, s = self.H[layer - 1]._get_best_matching(X[j,np.newaxis])
                    p = self.H[layer - 1].G.node[b]['pos']
                    e = np.hstack((e,self.H[layer - 1].G.node[b]['lab']))
                x = np.hstack((x,p))
            Xt[i+1-ws, :] = x
            c = collections.Counter(e)
            if c.most_common()[0][1] >= ws * lab_ratio:
                yt[i+1-ws] = c.most_common()[0][0]
            else:
                yt[i + 1 - ws] = -1
        return Xt, yt

    def _get_activation_trajectories(self, X, layer, y):

        for i in range(layer+1):
            X, y = self._get_trajectories_from_previous(X,i, y = y)
        return X, y

    def train(self, X, y):
        for k in range(self.layers + 1):
            X, y = self._get_activation_trajectories(X, i, y)
            self.H[k].train(X, y)

    def train_test_2L(self, X, y):

        X, y = self._get_activation_trajectories(X, 0, y)
        self.H[0].train(X, y, n_epochs=20)






#############################################################################

# Old test for propagating the labels through the whole hierarchy

from __future__ import division

import unittest
import numpy as np
from oss_gwr.oss_h import oss_h_unimodal

class TestOssGwrHFunctions(unittest.TestCase):

    def test_activation_trajectories(self):
        X = np.array([[1],
                      [2],
                      [3],
                      [4],
                      [5],
                      [6]])

        y = np.array([1,1,2,2,3,3])


        ossh = oss_h_unimodal(window_size=[1,2,3], lab_ratio=2/3)
        ossh.H[0].G.add_node(0, attr_dict={'pos': np.array([1]),
                                           'lab': 1})
        ossh.H[0].G.add_node(1, attr_dict={'pos': np.array([4]),
                                           'lab': 2})
        ossh.H[0].G.add_node(2, attr_dict={'pos': np.array([7]),
                                           'lab': 3})
        ossh.H[1].G.add_node(0, attr_dict={'pos': np.array([1, 1]),
                                           'lab': 1})
        ossh.H[1].G.add_node(1, attr_dict={'pos': np.array([5, 7]),
                                           'lab': 2})

        tr0_fp, _  = ossh._get_trajectories_from_previous(X,0,y)
        tr0, tr0_y = ossh._get_activation_trajectories(X,0,y)
        tr0_test   = X
        tr0_y_t    = y
        np.testing.assert_array_equal(tr0, tr0_test)
        np.testing.assert_array_equal(tr0_y, tr0_y_t)

        tr1_fp, _   = ossh._get_trajectories_from_previous(tr0,1)
        tr1, tr1_y  = ossh._get_activation_trajectories(X,1,y)
        tr1_test = np.array([[1, 1],
                             [1, 4],
                             [4, 4],
                             [4, 4],
                             [4, 7]])
        tr1_y_t = np.array([1, -1, 2, 2, -1])
        np.testing.assert_array_equal(tr1_fp, tr1)
        np.testing.assert_array_equal(tr1, tr1_test)
        np.testing.assert_array_equal(tr1_y, tr1_y_t)

        tr2_fp, _   = ossh._get_trajectories_from_previous(tr1, 2)
        tr2, tr2_y  = ossh._get_activation_trajectories(X,2,y)
        tr2_test = np.array([[1, 1, 1, 1, 5, 7],
                             [1, 1, 5, 7, 5, 7],
                             [5, 7, 5, 7, 5, 7]])
        tr2_y_t  = np.array([1, 2, 2])
        np.testing.assert_array_equal(tr2_fp,tr2)
        np.testing.assert_array_equal(tr2,tr2_test)
        np.testing.assert_array_equal(tr2_y, tr2_y_t)
