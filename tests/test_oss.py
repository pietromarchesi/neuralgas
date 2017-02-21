from __future__ import division

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
        gwr.G.add_node(0,attr_dict={'pos' : np.array([1, 1])})
        gwr.G.add_node(3,attr_dict={'pos' : np.array([2, 1])})

        b, s = gwr._get_best_matching(np.array([[3, 1]]))
        np.testing.assert_equal(b, 3)
        np.testing.assert_equal(s, 0)

        b, s = gwr._get_best_matching(np.array([[1, 2]]))
        np.testing.assert_equal(b, 0)
        np.testing.assert_equal(s, 3)


    def test_add_node(self):
        gwr = oss_gwr(fir_thr=1.1, act_thr=1.1)
        gwr.G = nx.Graph()
        gwr.G.add_node(0,attr_dict={'pos' : np.array([0, 0]),
                                    'fir' : 1,
                                    'lab'  : -1})
        gwr.G.add_node(3,attr_dict={'pos' : np.array([1, 0]),
                                    'fir' : 1,
                                    'lab' : -1})
        gwr._make_link(0,3)
        gwr._training_step(np.array([[4,0]]))
        np.testing.assert_equal(gwr.G.nodes()[-1], 4)
        pos = gwr.G.node[4]['pos']
        np.testing.assert_array_equal(pos, np.array([2.5,0]))
        np.testing.assert_array_equal(gwr.G.edges(),[(0,4),(3,4)])


    def test_label_propagation_coeff(self):
        gwr = oss_gwr()
        gwr.G = nx.Graph()
        gwr.G.add_node(0,attr_dict={'pos' : np.array([0, 0]),
                                    'fir' : 1,
                                    'lab'  : -1})
        gwr.G.add_node(1,attr_dict={'pos' : np.array([1, 0]),
                                    'fir' : 1,
                                    'lab' : -1})
        gwr._make_link(0,1)
        coeff = gwr._get_label_propagation_coeff(0,1)
        np.testing.assert_equal(coeff, 1/3)

        gwr.G.remove_edge(0,1)
        coeff = gwr._get_label_propagation_coeff(0, 1)
        np.testing.assert_equal(coeff, 0)


    def test_update_network(self):
        gwr = oss_gwr(eps_b=1,eps_n=1)
        gwr.G = nx.Graph()
        gwr.G.add_node(0,attr_dict={'pos' : np.array([0, 0]),
                                    'fir' : 1,
                                    'lab'  : -1})
        gwr.G.add_node(1,attr_dict={'pos' : np.array([1, 0]),
                                    'fir' : 1,
                                    'lab' : -1})
        gwr._make_link(0,1)
        gwr._update_network(np.array([[0, 1]]), 0)
        pos_b = gwr.G.node[0]['pos']
        pos_n = gwr.G.node[1]['pos']
        np.testing.assert_array_equal(pos_b, np.array([0, 1]))
        np.testing.assert_array_equal(pos_n, np.array([0, 1]))


    def test_remove_old_edges(self):
        gwr = oss_gwr(eps_b=1,eps_n=1,max_age=15)
        gwr.G = nx.Graph()
        gwr.G.add_node(0,attr_dict={'pos' : np.array([0, 0]),
                                    'fir' : 1,
                                    'lab'  : -1})
        gwr.G.add_node(1,attr_dict={'pos' : np.array([1, 0]),
                                    'fir' : 1,
                                    'lab' : -1})
        gwr.G.add_node(2,attr_dict={'pos' : np.array([1, 0]),
                                    'fir' : 1,
                                    'lab' : -1})
        gwr.G.add_node(3,attr_dict={'pos' : np.array([1, 0]),
                                    'fir' : 1,
                                    'lab' : -1})
        gwr.G.add_edge(0, 1, attr_dict={'age':20})
        gwr.G.add_edge(0, 3, attr_dict={'age':1})
        gwr.G.add_edge(1, 3, attr_dict={'age':1})
        gwr.G.add_edge(2, 3, attr_dict={'age':20})

        gwr._remove_old_edges()
        nodes = gwr.G.nodes()
        edges = gwr.G.edges()
        np.testing.assert_array_equal(nodes, [0, 1, 3])
        np.testing.assert_array_equal(edges, [(0, 3), (1, 3)])

        gwr.G.edge[1][3]['age'] = 30

        gwr._remove_old_edges()
        nodes = gwr.G.nodes()
        edges = gwr.G.edges()
        np.testing.assert_array_equal(nodes, [0, 3])
        np.testing.assert_array_equal(edges, [(0, 3)])

    def test_update_label(self):
        gwr = oss_gwr(eps_b=1,eps_n=1,max_age=15)
        gwr.G = nx.Graph()
        gwr.G.add_node(0,attr_dict={'pos' : np.array([0, 0]),
                                    'fir' : 0.1,
                                    'lab' : 3})
        gwr.G.add_node(1,attr_dict={'pos' : np.array([1, 0]),
                                    'fir' : 0.1,
                                    'lab' : 2})

        gwr._update_label(0,-1,1)
        np.testing.assert_equal(gwr.G.node[0]['lab'], 3)

        gwr._update_label(0,1,1)
        np.testing.assert_equal(gwr.G.node[0]['lab'], 1)

        gwr._update_label(0, -1, 1)
        np.testing.assert_equal(gwr.G.node[0]['lab'], 1)

        gwr.G.add_edge(0,1)
        gwr._update_label(0, -1, 1)
        np.testing.assert_equal(gwr.G.node[0]['lab'], 2)

