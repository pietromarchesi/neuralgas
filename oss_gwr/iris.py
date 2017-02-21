from __future__ import division

import logging
import numpy as np
import sklearn.metrics
import seaborn as sns
import sklearn.datasets
from oss_gwr.oss import oss_gwr
import matplotlib.pyplot as plt

iris = sklearn.datasets.load_iris()
X = iris.data[:,:2]
y = iris.target

ind = np.random.choice(X.shape[0], size = 100, replace=False)
mask = np.zeros_like(y,dtype = bool)
mask[ind] = True
X_train = X[mask,:]
y_train = y[mask]

X_test = X[~mask,:]
y_test = y[~mask]

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
oss = oss_gwr(act_thr=0.35)
oss.train(X_train, y_train, n_epochs = 20)
y_pred = oss.predict(X_test)

cm = sklearn.metrics.confusion_matrix(y_test, y_pred)

f,ax = plt.subplots(1,1)
sns.heatmap(cm,ax=ax,annot=True)
