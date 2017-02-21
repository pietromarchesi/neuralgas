import numpy as np
import logging
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

from oss_gwr.oss import oss_gwr

iris = sklearn.datasets.load_iris()
X = iris.data
y = iris.target

kf = sklearn.model_selection.KFold(n_splits=10)

acc = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # here you can kill the labels of y_train

    gwr = oss_gwr()
    gwr.train(X_train, y_train)
    y_pred = gwr.predict(X_test)
    a = sklearn.metrics.accuracy_score(y_test,y_pred)
    acc.append(a)

print('Mean accuracy: %s' %np.mean(acc))

