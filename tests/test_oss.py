import unittest
import numpy as np
import networkx as nx
from oss_gwr.oss import oss_gwr

class TestOssGwrFunctions(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[1, 1],
                           [2, 1],
                           [3, 1],
                           [4, 1]])


    def test_get_best_matching(self):

        gwr = oss_gwr()

        gwr.G = nx.Graph()
        gwr.G.add_node(0,attr_dict={'pos' : self.X[0,:]})
        gwr.G.add_node(1,attr_dict={'pos' : self.X[1,:]})
        b, s = gwr._get_best_matching(np.array([[3, 1]]))
        np.testing.assert_equal(b, 1)
        np.testing.assert_equal(s, 0)

        b, s = gwr._get_best_matching(np.array([[1, 2]]))
        np.testing.assert_equal(b, 0)
        np.testing.assert_equal(s, 1)

