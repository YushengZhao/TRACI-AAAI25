import numpy as np
import scipy.io as sio
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops


def load_citation_network(name, args):
    dat = sio.loadmat("./data/%s.mat" % name)
    attr = torch.tensor(dat['attrb'].toarray(), dtype=torch.float32)
    labels = torch.tensor(dat['group'])
    edges = torch.tensor(np.stack([dat['network'].tocoo().row, dat['network'].tocoo().col], axis=0), dtype=torch.long)
    edge_index, edge_attr = add_self_loops(edges)
    graph = Data(x=attr, edge_index=edge_index, edge_attr=edge_attr, y=labels).to(args.device)
    graph.name = name

    return graph
