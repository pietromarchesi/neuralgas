from __future__ import division

import unittest
import numpy as np
from neuralgas.oss_gwr_h import gwr_h_unimodal

class TestOssGwrHFunctions(unittest.TestCase):

    def test_activation_trajectories(self):
        X = np.array([[1],
                      [2],
                      [3],
                      [4],
                      [5],
                      [6]])

        ossh = gwr_h_unimodal(window_size=[1, 2, 3], lab_ratio=2 / 3)
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

        tr0_fp  = ossh._get_trajectories_from_previous(X,0)
        tr0     = ossh._get_activation_trajectories_old(X, 0)
        tr0_test   = X
        np.testing.assert_array_equal(tr0_fp, tr0)
        np.testing.assert_array_equal(tr0, tr0_test)

        tr1_fp   = ossh._get_trajectories_from_previous(tr0,1)
        tr1  = ossh._get_activation_trajectories_old(X, 1)
        tr1_test = np.array([[1, 1],
                             [1, 4],
                             [4, 4],
                             [4, 4],
                             [4, 7]])
        np.testing.assert_array_equal(tr1_fp, tr1)
        np.testing.assert_array_equal(tr1, tr1_test)

        tr2_fp   = ossh._get_trajectories_from_previous(tr1, 2)
        tr2  = ossh._get_activation_trajectories_old(X, 2)
        tr2_test = np.array([[1, 1, 1, 1, 5, 7],
                             [1, 1, 5, 7, 5, 7],
                             [5, 7, 5, 7, 5, 7]])
        np.testing.assert_array_equal(tr2_fp,tr2)
        np.testing.assert_array_equal(tr2,tr2_test)



