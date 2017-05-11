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

# prepare training data
a1_raw = pd.read_csv(dir + 'a1_raw.csv')
a1_va3 = np.genfromtxt(dir + 'a1_va3.csv', delimiter=',')
y_train = np.array(a1_raw['phase'].astype('category').cat.codes)
pos_train = np.array(a1_raw.iloc[:,:18])
dif = a1_raw.shape[0] - a1_va3.shape[0]
pos_train = pos_train[dif // 2 : -dif // 2, :18]
y_train = y_train[dif // 2 : -dif // 2]
vel_train = a1_va3[:,:32]
# select only the vectorial velocity of hands and wrists
vel_train = vel_train[:,0:12]
# reduce to binary classification of rest vs gesture
y_train[y_train != 2] = 1

# prepare test data
a2_raw = pd.read_csv(dir + 'a2_raw.csv')
a2_va3 = np.genfromtxt(dir + 'a2_va3.csv', delimiter=',')
y_test = np.array(a2_raw['phase'].astype('category').cat.codes)
pos_test = np.array(a2_raw.iloc[:,:18])
dif_test = a2_raw.shape[0] - a2_va3.shape[0]
pos_test = pos_test[dif // 2 : -dif // 2, :18]
y_test = y_test[dif // 2 : -dif // 2]
vel_test = a2_va3[:,:32]
# select only the vectorial velocity of hands and wrists
vel_test = vel_test[:,0:12]
# reduce to binary classification of rest vs gesture
y_test[y_test != 2] = 1


pars = {'act_thr' : 0.75, 'max_size':200}
unimodal_window = [5, 5]
STS_window = 5

def train_unsupervised_hierarchy(X1, X2, n_epochs = 30):
    g1 = gwr_h_unimodal(n_layers=2, window_size=unimodal_window, gwr_pars=pars)
    g2 = gwr_h_unimodal(n_layers=2, window_size=unimodal_window, gwr_pars=pars)
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
    X4   = _propagate_trajectories(X3_, ws=STS_window)
    return X4


def sequence_labels(y, w = 7):
    n = w-1
    y_ = -np.ones(y.shape[0] - n)
    for i in range(n,y.shape[0]):
        c = collections.Counter(y[i-n:i])
        lab = c.most_common()[0][0]
        y_[i-n] = lab
    return y_


n_epochs = 20
ran = range(10,110,10)
compare = True
N = 20

acc, ac2 = np.zeros([N, len(ran)]), np.zeros([N,len(ran)])

for j in range(N):
    for k in range(len(ran)):
        i = ran[k]
        y_train_ = sequence_labels(y_train, w=13)
        y_test_ = sequence_labels(y_test, w=13)

        g1, g2, g3 = train_unsupervised_hierarchy(pos_train,  vel_train, n_epochs=n_epochs)
        X_train = propagate_through_hierarchy(pos_train, vel_train, g1, g2, g3)
        X_test  = propagate_through_hierarchy(pos_test, vel_test, g1, g2, g3)

        gwr_super = oss_gwr(**pars)

        s = i * X_train.shape[0] // 100
        ind2 = np.random.choice(y_train_.shape[0], size=s, replace=False)
        mask = np.zeros_like(y_train_, dtype=bool)
        mask[ind2] = True
        y_train_[~mask] = -1

        gwr_super.train(X_train, y_train_, n_epochs=n_epochs)

        y_pred = gwr_super.predict(X_test)
        a = sklearn.metrics.accuracy_score(y_test_,y_pred)
        acc[j,k] = a

        if compare:
            pos_train_ = _propagate_trajectories(pos_train, ws=5)
            vel_train_ = _propagate_trajectories(vel_train, ws=5)
            X_train_ = np.hstack((pos_train_, vel_train_))
            X_train_ = _propagate_trajectories(X_train_, ws=5)
            X_train_ = _propagate_trajectories(X_train_, ws=5)

            pos_test_ = _propagate_trajectories(pos_test, ws=5)
            vel_test_ = _propagate_trajectories(vel_test, ws=5)
            X_test_ = np.hstack((pos_test_, vel_test_))
            X_test_ = _propagate_trajectories(X_test_, ws=5)
            X_test_ = _propagate_trajectories(X_test_, ws=5)

            gwr_super = oss_gwr(**pars)
            gwr_super.train(X_train, y_train_, n_epochs=n_epochs)

            y_pred = gwr_super.predict(X_test)
            a = sklearn.metrics.accuracy_score(y_test_, y_pred)
            ac2[j,k] = a

Acc = acc.mean(axis=0)
Ac2 = ac2.mean(axis=0)
ind = np.arange(len(ran))

f, ax = plt.subplots(1,1, figsize = [8,6])
ax.errorbar(ind, Acc, yerr=acc.std(axis=0), label = 'Hierarchical')
ax.errorbar(ind, Ac2, yerr=ac2.std(axis=0), label = 'Single gas')
ax.set_xticks(ind)
ax.set_xticklabels(ran)
# ax.legend()
ax.set_title('Classification accuracy')
ax.set_xlabel('% of labelled samples')
plt.legend()
#f.savefig('./plots/hierarchical_vs_single_gas_20epochs_20iters.eps',dpi=1000)
# TODO stop training based on some convergence criterion
