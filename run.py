# coding=utf-8
import numpy as np
import setproctitle
import importlib
from loguru import logger

from gnn.utils import *

args = get_args()
args.device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
setproctitle.setproctitle(args.expid)
device = args.device
encoder_dim = args.encoder_dim
data_loading_fn = importlib.import_module('gnn.dataset.%s' % args.experiment).load_data
TrainerClass = importlib.import_module('gnn.trainer.%s' % args.method).Trainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(args.seed)


if args.experiment == 'citation':
    source_graphs = [data_loading_fn(name, args) for name in args.source_names]
    target_graph = data_loading_fn(args.target_name, args)
    args.clf = 'multi'
    args.num_classes = target_graph.y.shape[1]
elif args.experiment == 'protein':
    proteins = data_loading_fn(args)
    target_idx = int(args.target_name)
    source_idx = [int(name) for name in args.source_names]
    target_graph = proteins[target_idx]
    source_graphs = [proteins[i] for i in source_idx]
    args.clf = 'single'
    args.num_classes = 8

args.num_input_feat = target_graph.x.shape[1]


def main():
    args.source_graphs = source_graphs
    args.target_graph = target_graph
    trainer = TrainerClass(args)

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
