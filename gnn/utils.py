import random
from torch_geometric.data import Data
import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse


class Buffer:
    def __init__(self, size):
        self.buffer_size = size
        self.buf = []

    def add(self, obj):
        if len(self.buf) >= self.buffer_size:
            self.buf.pop()
        self.buf.append(obj)

    def sample(self, n):
        result = random.choices(self.buf, k=n)
        return result


def get_normalized_lap(edge_index, num_nodes):
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0] 
    adj = ((adj + torch.eye(adj.size(0), device=adj.device)) > 0).float()
    sum_0 = torch.sum(adj, dim=0, keepdim=True)
    sum_1 = torch.sum(adj, dim=1, keepdim=True)
    deg = torch.diag(torch.sum(adj, dim=0))
    normalized_lap = ((deg - adj) / torch.sqrt(sum_0)) / torch.sqrt(sum_1)
    return normalized_lap.to_sparse()


def dropout_edge(edge_index, p=0.5,
                 force_undirected=False,
                 training=True):
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask


def graph_augmentation(graph, edge_drop_prob=0.3):
    aug_edge_index, _ = dropout_edge(graph.edge_index, p=edge_drop_prob, force_undirected=True)
    aug_graph = Data(
        x=graph.x.clone().detach(),
        edge_index=aug_edge_index,
        edge_attr=graph.edge_attr.clone().detach() if graph.edge_attr is not None else None,
        y=graph.y.clone().detach(),
        name=graph.name
    )
    if edge_drop_prob > 0:
        aug_graph.lap = get_normalized_lap(aug_edge_index, aug_graph.num_nodes)
    else:
        aug_graph.lap = graph.lap.clone().detach()
    return aug_graph


def compute_smoothness(lap, x):
    latter = torch.sparse.mm(lap, x)
    result = torch.sum(x * latter)
    return result


def evaluate_multi(preds, labels):  # N x C
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()

    tp = labels * preds
    fp = (1 - labels) * preds
    fn = labels * (1 - preds)
    # macro f1:
    macro_tp = tp.sum(dim=0)
    macro_fp = fp.sum(dim=0)
    macro_fn = fn.sum(dim=0)
    precision = macro_tp / (macro_tp + macro_fp)
    recall = macro_tp / (macro_tp + macro_fn)
    f1 = 2 * precision * recall / (precision + recall)
    macro_f1 = f1.mean()

    # micro f1:
    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    precision = micro_tp / (micro_tp + micro_fp)
    recall = micro_tp / (micro_tp + micro_fn)
    micro_f1 = 2 * precision * recall / (precision + recall)
    return micro_f1, macro_f1, accuracy, torch.tensor(0, device=accuracy.device)


def evaluate_single(preds, labels, probs=None):
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    if probs is not None and torch.max(labels) == 1:
        from sklearn import metrics
        y = labels.cpu().numpy()
        probs = probs.cpu().numpy()
        fpr, tpr, thresholds = metrics.roc_curve(y, probs, pos_label=1)
        auc = torch.tensor(metrics.auc(fpr, tpr), device=accuracy.device)
    else:
        auc = torch.tensor(0, device=accuracy.device)
    return torch.tensor(0, device=accuracy.device), torch.tensor(0, device=accuracy.device), accuracy, auc


def ema_update(target_model, source_model, beta=0.99):
    with torch.no_grad():
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(beta * target_param.data + (1.0 - beta) * source_param.data)

def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--encoder_dim", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--gconv_type", type=str)
    parser.add_argument("--input_init_pert", type=float)
    parser.add_argument("--hidden_init_pert", type=float)
    parser.add_argument("--input_smooth_loss", type=float)
    parser.add_argument("--hidden_smooth_loss", type=float)
    parser.add_argument("--input_adv_lr", type=float)
    parser.add_argument("--hidden_adv_lr", type=float)
    parser.add_argument("--adv_max_iters", type=int)
    parser.add_argument("--buffer_size", type=int)
    parser.add_argument("--resample", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--edge_drop_prob", type=float)
    parser.add_argument("--beta_param1", type=float)
    parser.add_argument("--beta_param2", type=float)
    parser.add_argument("--cl_loss_weight", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--warmup_epochs", type=int, default=150)
    parser.add_argument("--experiment", type=str, choices=['protein', 'citation'], default='citation')
    parser.add_argument("--method", type=str, choices=['traci'], default='traci')
    parser.add_argument("--source-names", type=str, nargs='+')
    parser.add_argument("--target-name", type=str)

    args = parser.parse_args()