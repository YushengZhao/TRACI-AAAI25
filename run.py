# coding=utf-8
from argparse import ArgumentParser

import numpy as np
import setproctitle
from loguru import logger

from gnn.dataset.citation import load_citation_network
from gnn.dataset.protein import load_protein_networks
from gnn.utils import *
from train_methods.traci import TraciTrainer

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

args = parser.parse_args()
args.device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
setproctitle.setproctitle(args.expid)
device = args.device
encoder_dim = args.encoder_dim


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(args.seed)


if args.experiment == 'citation':
    # acmv9 citationv1 dblpv7
    source_graphs = [
        load_citation_network("acmv9", args),
        load_citation_network("dblpv7", args)
    ]
    target_graph = load_citation_network("citationv1", args)
    args.clf = 'multi'
    args.num_classes = target_graph.y.shape[1]
elif args.experiment == 'protein':
    proteins = load_protein_networks(args)
    target_idx = 2
    source_idx = [0, 1, 3]
    target_graph = proteins[target_idx]
    source_graphs = [proteins[i] for i in source_idx]
    args.clf = 'single'
    args.num_classes = 8

args.num_input_feat = target_graph.x.shape[1]


def main():
    args.source_graphs = source_graphs
    args.target_graph = target_graph
    trainer = TraciTrainer(args)

    for epoch in range(1, args.epochs + 1):
        stats = trainer.train(epoch, source_graphs)
        (test_micro_f1, test_macro_f1, test_acc, test_auc), test_loss = trainer.test(target_graph, target_graph.name, output_loss=True)
        if args.clf == 'multi':
            logger.info("Epoch %d: TGT_MICRO %.4f, TGT_MACRO %.4f, TGT_ACC %.4f, %s "
                        % (epoch, test_micro_f1, test_macro_f1, test_acc, stats))
        else:
            logger.info("Epoch %d: TGT_ACC: %.4f, TGT_AUC %.4f" % (epoch, test_acc, test_auc))


if __name__ == '__main__':
    main()
