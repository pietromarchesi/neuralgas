from __future__ import division

import copy
import logging
import numpy as np
import networkx as nx
import collections
from neuralgas.oss_gwr import gwr


class gwr_h_unimodal():

    def __init__(self, n_layers = 2, window_size = [3, 1], lab_ratio =1 / 2,
                 gwr_pars = []):
        '''
        NOTE: The data X is used in training without window for layer 1,
        then window_size[0] is used to get the trajectory from the first
        to the second layer. If you have only two layers, window_size[1]
        is the output trajectory already windowed.
        '''

        self.n_layers    = n_layers
        self.window_size = window_size
        self.lab_ratio   = lab_ratio

        if isinstance(gwr_pars, list):
            self.H = []
            for i in range(self.n_layers):
                try:
                    par = gwr_pars[i]
                    self.H.append(gwr(**par))
                except IndexError:
                    self.H.append(gwr())
        elif isinstance(gwr_pars, dict):
            self.H = [gwr(**gwr_pars) for _ in range(self.n_layers)]


    def _get_activation_trajectories(self, X):

        for j in range(self.n_layers):
            X = _propagate_trajectories(X, network= self.H[j],
                                        ws = self.window_size[j])
        return X


    def train(self, X, n_epochs = 20):

        if hasattr(n_epochs, '__len__'):
            if len(n_epochs) == self.n_layers:
                n_ep = n_epochs
            else:
                raise ValueError('Array of epoch numbers is not of the'
                                 'correct size.')
        else:
            n_ep = [n_epochs]*self.n_layers

        for i in range(self.n_layers):
            self.H[i].train(X, n_epochs=n_ep[i])
            X = _propagate_trajectories(X, network= self.H[i],
                                        ws=self.window_size[i])
        return X


def _propagate_trajectories(X, network = None, ws = 3):
    Xt = np.zeros([X.shape[0]-(ws-1), X.shape[1]*ws])

    for i in range((ws-1), X.shape[0]):
        x = np.zeros([0,])
        for j in range(i+1-ws,i+1):
            if network is None:
                p = X[j,:]
            else:
                b, s = network._get_best_matching(X[j, np.newaxis])
                p = network.G.node[b]['pos']
            x = np.hstack((x,p))
        Xt[i+1-ws, :] = x
    return Xt

# TODO introduce a random seed to initialize the


class gwr_h_bimodal():
    # still unsupervised

    def train(self, X1, X2):
        pass
