import torch
import torch.nn.functional as F
from torch import nn

from gnn.cached_gcn_conv import CachedGCNConv
from gnn.gconv import *


class GNN(torch.nn.Module):
    def __init__(self, args, base_model=None, type="gcn", **kwargs):
        super(GNN, self).__init__()
        self.args = args
        if base_model is None:
            weights = [None, None]
            biases = [None, None]
        else:
            weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias for conv_layer in base_model.conv_layers]

        self.dropout_layers = [nn.Dropout(0.1) for _ in weights]
        self.type = type

        if type == 'gcn':
            model_cls = CachedGCNConv
        elif type == 'sage':
            model_cls = SAGEConvolution
        elif type == 'gin':
            model_cls = GINConvolution
        elif type == 'gat':
            model_cls = GATConvolution
        elif type == 'appnp':
            model_cls = APPNPConvolution
        elif type == 'sgc':
            model_cls = SGConvolution
        elif type == 'egc':
            model_cls = EGConvolution
        else:
            model_cls = None

        if type in ['ppmi', 'gcn', 'sage', 'gin', 'gat', 'egc']:
            self.conv_layers = nn.ModuleList([
                model_cls(args, args.num_input_feat, args.hidden_dim,
                          weight=weights[0],
                          bias=biases[0],
                          **kwargs),
                model_cls(args, args.hidden_dim, args.encoder_dim,
                          weight=weights[1],
                          bias=biases[1],
                          **kwargs)
            ])
        elif type == 'appnp':
            self.conv_layers = nn.ModuleList([
                APPNPConvolution(gnn.MLP([args.num_input_feat, args.hidden_dim, args.encoder_dim]))
            ])
        elif type == 'sgc':
            self.conv_layers = nn.ModuleList([
                SGConvolution(args.num_input_feat, args.encoder_dim)
            ])

    def forward(self, x, edge_index, cache_name, edge_weight=None, hidden_noise=None,
                # args for prototype computation
                compute_prototype=False, graph=None, prototype_aug=None):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, cache_name, edge_weight)
            x = F.relu(x)
            x = self.dropout_layers[i](x)
            if i == 0:
                if compute_prototype:
                    prototypes_mean = torch.zeros((self.args.num_classes, x.size(-1)), device=self.args.device)  # Nc x D
                    prototypes_std = torch.zeros_like(prototypes_mean)
                    if self.args.experiment == 'citation':
                        labels = graph.y  # N x Nc
                        for cls in range(self.args.num_classes):
                            mask = (labels[:, cls] > 0)
                            if mask.int().sum() > 0:
                                cls_feat = x[mask, :]
                                prototypes_mean[cls, :] = cls_feat.mean(dim=0)
                                prototypes_std[cls, :] = cls_feat.std(dim=0)
                    elif self.args.experiment in ['protein', 'twitch']:
                        labels = graph.y
                        for cls in range(self.args.num_classes):
                            mask = (labels == cls)
                            if mask.int().sum() > 0:
                                cls_feat = x[mask, :]
                                prototypes_mean[cls, :] = cls_feat.mean(dim=0)
                                prototypes_std[cls, :] = cls_feat.std(dim=0, correction=0)
                    else:
                        raise ValueError("Unknown experiment")
                    graph.prototypes = (prototypes_mean.clone().detach(), prototypes_std.clone().detach())
                if prototype_aug is not None:
                    aug_beta = torch.distributions.Beta(torch.tensor([5.0], device=self.args.device),
                                                        torch.tensor([1.0], device=self.args.device))
                    aug_ratio = aug_beta.sample((x.size(0), ))
                    x = x * aug_ratio + prototype_aug * (1 - aug_ratio)
                if hidden_noise is not None:
                    x = x + hidden_noise
        return x
