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

        self.H = [oss_gwrd() for _ in range(self.layers)]
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






