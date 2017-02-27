from __future__ import division

import numpy as np
import pandas as pd
import sklearn.model_selection
import collections
import matplotlib.pyplot as plt
from neurodecoder import RandomForestDecoder
from neuralgas.oss_gwr_h import gwr_h_unimodal
from neuralgas.oss_gwr_h import _propagate_trajectories
from neuralgas.oss_gwr import gwr
from neuralgas.oss_gwr import oss_gwr

dir = '/home/pietro/data/gesture_phase_dataset/'

a1_raw = pd.read_csv(dir + 'a1_raw.csv')
a1_va3 = np.genfromtxt(dir + 'a1_va3.csv', delimiter=',')

y = np.array(a1_raw['phase'].astype('category').cat.codes)

pos = np.array(a1_raw.iloc[:,:18])
dif = a1_raw.shape[0] - a1_va3.shape[0]
pos = pos[dif // 2 : -dif // 2, :18]
y = y[dif // 2 : -dif // 2]

vel = a1_va3[:,:32]

pars = {'act_thr' : 0.75, 'max_size':200}


def train_unsupervised_hierarchy(X1, X2, n_epochs = 30):
    g1 = gwr_h_unimodal(n_layers=2, window_size=[3, 3], gwr_pars=pars)
    g2 = gwr_h_unimodal(n_layers=2, window_size=[3, 3], gwr_pars=pars)
    g3 = gwr(**pars)

    X1_ = g1.train(X1, n_epochs=n_epochs)
    X2_ = g2.train(X2, n_epochs=n_epochs)
    X   = np.hstack((X1_, X2_))

    g3.train(X, n_epochs=n_epochs)
    return g1, g2, g3

def propagate_through_hierarchy(X1, X2, g1, g2, g3):
    X1_  = g1._get_activation_trajectories(X1)
    X2_  = g2._get_activation_trajectories(X2)
    X3   = np.hstack((X1_, X2_))
    X3_  = _propagate_trajectories(X3, g3, ws=1)
    X4   = _propagate_trajectories(X3_, ws=3)
    return X4


def sequence_labels(y, w = 7):
    n = w-1
    y_ = -np.ones(y.shape[0] - n)
    for i in range(n,y.shape[0]):
        c = collections.Counter(y[i-n:i])
        lab = c.most_common()[0][0]
        y_[i-n] = lab
    return y_

acc = []
accrf = []
compare = False
n_splits = 6

kf = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=False)
for train_index, test_index in kf.split(pos):
    pos_train, pos_test = pos[train_index], pos[test_index]
    vel_train, vel_test = vel[train_index], vel[test_index]

    y_train, y_test = y[train_index], y[test_index]
    y_train = sequence_labels(y_train, w=7)
    y_test = sequence_labels(y_test, w=7)

    g1, g2, g3 = train_unsupervised_hierarchy(pos_train,  vel_train, n_epochs=40)
    X_train = propagate_through_hierarchy(pos_train, vel_train, g1, g2, g3)
    X_test  = propagate_through_hierarchy(pos_test, vel_test, g1, g2, g3)

    gwr_super = oss_gwr(**pars)
    gwr_super.train(X_train, y_train, n_epochs=40)

    y_pred = gwr_super.predict(X_test)
    a = sklearn.metrics.accuracy_score(y_test,y_pred)
    acc.append(a)

    if compare:
        rf = RandomForestDecoder()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        a = sklearn.metrics.accuracy_score(y_test,y_pred)
        accrf.append(a)

f, ax = plt.subplots(1,1)
ax.plot(acc, label = 'GWR')
#ax.plot(accrf, label = 'RF')
ax.set_xticklabels(range(n_splits))
ax.legend()
ax.set_title('Classification accuracy')