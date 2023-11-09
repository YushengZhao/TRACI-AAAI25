import torch_geometric.nn as gnn


class SAGEConvolution(gnn.SAGEConv):
    def __init__(self, args, in_channel, out_channel, **kwargs):
        super(SAGEConvolution, self).__init__(in_channel, out_channel)

    def forward(self, x, edge_index, cache_name, edge_weight):
        return super(SAGEConvolution, self).forward(x, edge_index)


class GINConvolution(gnn.GINConv):
    def __init__(self, args, in_channel, out_channel, **kwargs):
        super(GINConvolution, self).__init__(
            gnn.MLP([in_channel, out_channel, out_channel])
        )
        
    def forward(self, x, edge_index, cache_name, edge_weight):
        return super(GINConvolution, self).forward(x, edge_index)


class GATConvolution(gnn.GATConv):
    def __init__(self, args, in_channel, out_channel, **kwargs):
        super(GATConvolution, self).__init__(in_channel, out_channel)
        
    def forward(self, x, edge_index, cache_name, edge_weight):
        return super(GATConvolution, self).forward(x, edge_index, edge_weight)


class APPNPConvolution(gnn.APPNP):
    def __init__(self, encoder):
        super(APPNPConvolution, self).__init__(2, 0.1)
        self.encoder = encoder

    def forward(self, x, edge_index, cache_name, edge_weight):
        x = self.encoder(x)
        return super(APPNPConvolution, self).forward(x, edge_index, edge_weight)


class SGConvolution(gnn.SGConv):
    def __init__(self, in_channel, out_channel):
        super(SGConvolution, self).__init__(in_channel, out_channel, 2)
    
    def forward(self, x, edge_index, cache_name, edge_weight):
        return super(SGConvolution, self).forward(x, edge_index, edge_weight)


class EGConvolution(gnn.EGConv):
    def __init__(self, args, in_channel, out_channel, **kwargs):
        super(EGConvolution, self).__init__(in_channel, out_channel, num_heads=1, num_bases=1)

    def forward(self, x, edge_index, cache_name, edge_weight):
        return super(EGConvolution, self).forward(x, edge_index)
