from __future__ import division

import numpy as np
import pandas as pd
import sklearn.model_selection
import collections
from neuralgas.oss_gwr_h import gwr_h_unimodal
from neuralgas.oss_gwr import oss_gwr

dir = '/home/pietro/data/gesture_phase_dataset/'

a1_raw = pd.read_csv(dir + 'a1_raw.csv')
a1_va3 = np.genfromtxt(dir + 'a1_va3.csv', delimiter=',')

y = np.array(a1_raw['phase'].astype('category').cat.codes)

pos = np.array(a1_raw.iloc[:,:18])
dif = a1_raw.shape[0] - a1_va3.shape[0]
pos = pos[dif // 2 : -dif // 2, :18]
y = y[dif // 2 : -dif // 2]

vel = a1_va3

pars = {'act_thr' : 0.75}

g1 = gwr_h_unimodal(n_layers=2,window_size=[3,3], gwr_pars=pars)
g2 = gwr_h_unimodal(n_layers=2, window_size=[3,3], gwr_pars=pars)

pos_ = g1.train(pos,n_epochs=20)
vel_ = g2.train(vel,n_epochs=20)

#posc = g1._get_activation_trajectories(pos)

X_ = np.hstack((pos_, vel_))

y_ = -np.ones(X_.shape[0])
for i in range(5,y.shape[0]):
    c = collections.Counter(y[i-5:i])
    lab = c.most_common()[0][0]
    y_[i-5] = lab

acc = []
kf = sklearn.model_selection.KFold(n_splits=10)
for train_index, test_index in kf.split(X_):
    X_train, X_test = X_[train_index], X_[test_index]
    y_train, y_test = y_[train_index], y_[test_index]
    # here you can kill the labels of y_train

    gwr = oss_gwr()
    gwr.train(X_train, y_train)
    y_pred = gwr.predict(X_test)
    a = sklearn.metrics.accuracy_score(y_test,y_pred)
    acc.append(a)

