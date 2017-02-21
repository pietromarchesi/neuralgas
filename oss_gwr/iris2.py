from __future__ import division

import logging
import numpy as np
import sklearn.metrics
import networkx as nx
import seaborn as sns
import sklearn.datasets
from oss_gwr.oss import oss_gwr
import matplotlib.pyplot as plt

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

iris = sklearn.datasets.load_iris()
X = iris.data
y = iris.target

oss = oss_gwr(act_thr=0.35)
oss.train(X, y, n_epochs = 20)
y_pred = oss.predict(X)

cm = sklearn.metrics.confusion_matrix(y, y_pred)
acc = cm.diagonal().sum() / cm.sum()
print('Classification accuracy: %s' %str(acc))

# f,ax = plt.subplots(1,1)
# sns.heatmap(cm,ax=ax,annot=True)
# plt.show()