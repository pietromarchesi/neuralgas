import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from oss_gwr.oss import oss_gwr

gwr = oss_gwr(tau_b=0.3,tau_n=0.1,kappa=1.05)
gwr.G = nx.Graph()
gwr.G.add_node(0, attr_dict={'pos': np.array([1,1]), 'fir' : 1})
gwr.G.add_node(1, attr_dict={'pos': np.array([1,1]), 'fir' : 1})
gwr.G.add_edge(0, 1)
fir, fir_n = [], []

for i in range(10):
    gwr._update_firing(0)
    fir.append(gwr.G.node[0]['fir'])
    fir_n.append(gwr.G.node[1]['fir'])

f, ax = plt.subplots(1,1)
ax.set_title('Synapse habituation')
ax.plot(fir, label='Best matching synapse')
ax.plot(fir_n, label='Neighbor synapse')
ax.legend()