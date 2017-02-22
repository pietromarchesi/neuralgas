from __future__ import division

import unittest
import numpy as np
from oss_gwr.oss_h import oss_h_unimodal

class TestOssGwrHFunctions(unittest.TestCase):

    def test_activation_trajectories(self):
        X = np.array([[1],
                      [2],
                      [3],
                      [4],
                      [5],
                      [6]])

        y = np.array([1,1,2,2,3,3])


        ossh = oss_h_unimodal(window_size=[1,2,3], lab_ratio=2/3)
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

        tr0_fp, _  = ossh._get_trajectories_from_previous(X,0,y)
        tr0, tr0_y = ossh._get_activation_trajectories(X,0,y)
        tr0_test   = X
        tr0_y_t    = y
        np.testing.assert_array_equal(tr0, tr0_test)
        np.testing.assert_array_equal(tr0_y, tr0_y_t)

        tr1_fp, _   = ossh._get_trajectories_from_previous(tr0,1)
        tr1, tr1_y  = ossh._get_activation_trajectories(X,1,y)
        tr1_test = np.array([[1, 1],
                             [1, 4],
                             [4, 4],
                             [4, 4],
                             [4, 7]])
        # the trajectories 1 and 4 do not have 2/3 of the labels which
        # are the same, hence they are not labelled
        tr1_y_t = np.array([1, -1, 2, 2, -1])
        np.testing.assert_array_equal(tr1_fp, tr1)
        np.testing.assert_array_equal(tr1, tr1_test)
        np.testing.assert_array_equal(tr1_y, tr1_y_t)

        tr2_fp, _   = ossh._get_trajectories_from_previous(tr1, 2)
        tr2, tr2_y  = ossh._get_activation_trajectories(X,2,y)
        tr2_test = np.array([[1, 1, 1, 1, 5, 7],
                             [1, 1, 5, 7, 5, 7],
                             [5, 7, 5, 7, 5, 7]])
        tr2_y_t  = np.array([1, 2, 2])
        np.testing.assert_array_equal(tr2_fp,tr2)
        np.testing.assert_array_equal(tr2,tr2_test)
        np.testing.assert_array_equal(tr2_y, tr2_y_t)
