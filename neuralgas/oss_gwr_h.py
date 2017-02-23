from __future__ import division

import logging
import numpy as np
import networkx as nx
import collections
from neuralgas.oss_gwr import gwr


class gwr_h_unimodal():

    def __init__(self, layers = 2, window_size = [1,3], lab_ratio = 1/2,
                 network_pars = []):
        '''
        NOTE: The data X is used in training without window for layer 1,
        then window_size[0] is used to get the trajectory from the first
        to the second layer. If you have only two layers, window_size[1]
        is the output trajectory already windowed.
        '''


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

        self.H = [gwr() for _ in range(self.layers)]
        # for i in range(self.layers):
        #     self.H[i].G = nx.Graph()

    # def _get_trajectories_from_previous(self, X, layer):
    #     ws = self.window_size[layer]
    #     Xt = np.zeros([X.shape[0]-(ws-1), X.shape[1]*ws])
    #     for i in range((ws-1), X.shape[0]):
    #
    #         x = np.zeros([0,])
    #         for j in range(i+1-ws,i+1):
    #             if layer == 0:
    #                 p = X[j,0]
    #             else:
    #                 b, s = self.H[layer - 1]._get_best_matching(X[j,np.newaxis])
    #                 p = self.H[layer - 1].G.node[b]['pos']
    #             x = np.hstack((x,p))
    #         Xt[i+1-ws, :] = x
    #     return Xt
    #
    def _get_activation_trajectories_old(self, X, layer):
        for i in range(layer+1):
            X = self._get_trajectories_from_previous(X,i)
        return X

    def _get_activation_trajectories(self, X):
        for i in range(self.layers):
            X = _propagate_trajectories(X, oss = self.H[0],
                                        ws = self.window_size[i])
        return X

    # def train(self, X, n_epochs = 20):
    #     for k in range(self.layers + 1):
    #         X = self._get_activation_trajectories(X, k)
    #         self.H[k].train(X, n_epochs=n_epochs)

    def train_test_2L(self, X, n_epochs=20):

        X = _propagate_trajectories(X, ws = 1)
        self.H[0].train(X, n_epochs=20)
        X1 = _propagate_trajectories(X, oss = self.H[0])
        self.H[1].train(X1, n_epochs=20)

    def train(self, X, n_epochs = 20):
        if hasattr(n_epochs, '__len__'):
            if len(n_epochs) == self.layers:
                n_ep = n_epochs
            else:
                raise ValueError('Array of epoch numbers is not of the'
                                 'correct size.')
        else:
            n_ep = [n_epochs]*self.layers

        for i in range(self.layers):
            self.H[i].train(X, n_epochs=n_ep[i])
            X = _propagate_trajectories(X, oss = self.H[i],
                                        ws=self.window_size[i])
        return X


def _propagate_trajectories(X, oss = None, ws = 3):
    Xt = np.zeros([X.shape[0]-(ws-1), X.shape[1]*ws])
    for i in range((ws-1), X.shape[0]):

        x = np.zeros([0,])
        for j in range(i+1-ws,i+1):
            if oss is None:
                p = X[j,:]
            else:
                b, s = oss._get_best_matching(X[j,np.newaxis])
                p = oss.G.node[b]['pos']
            x = np.hstack((x,p))
        Xt[i+1-ws, :] = x
    return Xt

# TODO introduce a random seed to initialize the


class gwr_h_multimodal():
    # still unsupervised

    def train(self, X):
        pass