from gnn.models import GNN
import torch.nn as nn
import itertools
import torch
from gnn.utils import evaluate_single, evaluate_multi, graph_augmentation, compute_smoothness
import torch.nn.functional as F
from gnn.utils import Buffer, get_normalized_lap


class TraciTrainer:
    def __init__(self, args, GNNcls=GNN):
        self.args = args
        self.device = args.device
        self.encoder = GNNcls(args, type=args.gconv_type).to(args.device)
        self.cls_model = nn.Linear(args.encoder_dim, args.num_classes).to(args.device)
        self.models = [self.encoder, self.cls_model]
        self.params = itertools.chain(*[model.parameters() for model in self.models])
        self.optimizer = torch.optim.Adam(self.params, lr=args.lr)
        self.ce_loss = nn.CrossEntropyLoss().to(args.device)
        self.bce_loss = nn.BCEWithLogitsLoss().to(args.device)
        self.preprocess_graph(args.target_graph)
        for graph in args.source_graphs:
            self.preprocess_graph(graph)

    def preprocess_graph(self, graph):
        graph.lap = get_normalized_lap(graph.edge_index, graph.num_nodes)
        graph.buf = Buffer(5)
        graph.prototypes = None

    def encode(self, data, cache_name, mask=None, noise=None, hidden_noise=None, **kwargs):
        if noise is None:
            encoded_output = self.encoder(data.x, data.edge_index, cache_name, data.edge_attr, hidden_noise, **kwargs)
        else:
            encoded_output = self.encoder(data.x + noise, data.edge_index, cache_name, data.edge_attr, hidden_noise,
                                     **kwargs)
        if mask is not None:
            encoded_output = encoded_output[mask]
        return encoded_output

    def predict(self, data, cache_name, mask=None, output_embedding=False):
        encoded_output = self.encode(data, cache_name, mask)
        logits = self.cls_model(encoded_output)
        if output_embedding:
            return logits, encoded_output
        else:
            return logits

    def compute_loss(self, logit, y):
        if self.args.clf == 'multi':
            return self.bce_loss(logit, y.float())
        else:
            return self.ce_loss(logit, y)

    def train(self, epoch, source_graphs):
        args = self.args
        models = self.models
        encode = self.encode
        cls_model = self.cls_model
        optimizer = self.optimizer
        compute_loss = self.compute_loss
        for model in models:
            model.eval()
        adv_pert_inputs = []
        adv_pert_hiddens = []
        aug_graphs = []
        if epoch > args.warmup_epochs:
            max_iters = ((epoch - args.warmup_epochs) * args.adv_max_iters) // (args.epochs - args.warmup_epochs) + 1
        else:
            max_iters = 0
        for graph in source_graphs:
            aug_graph = graph_augmentation(graph)
            input_noise = torch.rand_like(aug_graph.x) * args.input_init_pert * 2 - args.input_init_pert
            hidden_noise = torch.rand((aug_graph.num_nodes, args.hidden_dim), device=args.device) \
                           * args.hidden_init_pert * 2 - args.hidden_init_pert
            for i in range(max_iters):
                self.optimizer.zero_grad()
                input_noise.requires_grad = True
                hidden_noise.requires_grad = True
                feat = encode(aug_graph, None, noise=input_noise, hidden_noise=hidden_noise)
                logits = cls_model(feat)
                probs = torch.softmax(logits, dim=-1)
                loss_entropy = torch.sum(probs * torch.log(probs), dim=-1).mean()
                loss_smoothness_input = compute_smoothness(aug_graph.lap, input_noise) / aug_graph.num_nodes
                loss_smoothness_hidden = compute_smoothness(aug_graph.lap, hidden_noise) / aug_graph.num_nodes
                adv_loss = loss_entropy \
                           - loss_smoothness_input * args.input_smooth_loss \
                           - loss_smoothness_hidden * args.hidden_smooth_loss
                adv_loss.backward()
                input_grad = input_noise.grad.detach()
                hidden_grad = hidden_noise.grad.detach()
                input_noise.detach_()
                hidden_noise.detach_()
                input_noise -= args.input_adv_lr * torch.sign(input_grad)
                hidden_noise -= args.hidden_adv_lr * torch.sign(hidden_grad)
                input_noise.clamp_(min=-args.input_init_pert, max=args.input_init_pert)
                hidden_noise.clamp_(min=-args.hidden_init_pert, max=args.hidden_init_pert)
            adv_pert_inputs.append(input_noise.clone().detach())
            adv_pert_hiddens.append(hidden_noise.clone().detach())
            aug_graphs.append(aug_graph)
            graph.buf.add((input_noise.clone().detach(), hidden_noise.clone().detach(), aug_graph))
        cls_loss = 0
        cls_count = 0
        contrastive_loss = 0
        contrastive_count = 0
        optimizer.zero_grad()
        for model in models:
            model.train()
        for i in range(len(source_graphs)):
            original_feat = encode(source_graphs[i], None, compute_prototype=True, graph=source_graphs[i])
            original_logits = cls_model(original_feat)
            cls_loss += compute_loss(original_logits, source_graphs[i].y)
            cls_count += 1
            cur_adv_feat = encode(aug_graphs[i], None, noise=adv_pert_inputs[i], hidden_noise=adv_pert_hiddens[i])
            cur_adv_logits = cls_model(cur_adv_feat)
            cls_loss += compute_loss(cur_adv_logits, aug_graphs[i].y)
            cls_count += 1
            resampled_noise = source_graphs[i].buf.sample(3)
            for resampled_input, resampled_hidden, resampled_aug_graph in resampled_noise:
                resampled_feat = encode(resampled_aug_graph, None, noise=resampled_input,
                                        hidden_noise=resampled_hidden)
                resampled_logits = cls_model(resampled_feat)
                cls_loss += compute_loss(resampled_logits, resampled_aug_graph.y)
                cls_count += 1
            if epoch > args.warmup_epochs:
                domain_number = len(source_graphs)
                node_number = source_graphs[i].num_nodes
                dirichlet_alphas = torch.ones((domain_number,), device=args.device) / domain_number
                domain_dirichlet = torch.distributions.Dirichlet(dirichlet_alphas)
                aug_mean = torch.stack([graph.prototypes[0] for graph in source_graphs])
                aug_std = torch.stack([graph.prototypes[1] for graph in source_graphs])
                augmentation = torch.zeros((node_number, args.hidden_dim), device=args.device)
                domain_dist = domain_dirichlet.sample((node_number,))

                labels = source_graphs[i].y
                for cls in range(args.num_classes):
                    if args.experiment == 'citation':
                        mask = (labels[:, cls] > 0)
                    elif args.experiment in ['protein', 'twitch']:
                        mask = (labels == cls)
                    else:
                        raise ValueError("Unknown experiment")
                    gaussian = torch.randn((node_number, domain_number, args.hidden_dim),
                                           device=args.device)  # N x Nd x D
                    gaussian = gaussian * aug_std[:, cls, :] + aug_mean[:, cls, :]
                    gaussian = (gaussian * domain_dist.unsqueeze(-1)).sum(dim=1)  # N x D
                    gaussian = gaussian * mask.float().unsqueeze(-1)
                    augmentation = augmentation + gaussian
                pt_aug_feat = encode(source_graphs[i], None, prototype_aug=augmentation)
                pt_aug_logits = cls_model(pt_aug_feat)
                cls_loss += compute_loss(pt_aug_logits, source_graphs[i].y)
                cls_count += 1
                original_feat_norm = F.normalize(original_feat, p=2, dim=1)
                pt_aug_feat_norm = F.normalize(pt_aug_feat, p=2, dim=1)
                similarity = original_feat_norm @ pt_aug_feat_norm.T
                exp_similarity = torch.exp(similarity)
                positive = torch.diag(exp_similarity)
                negative = torch.sum(exp_similarity, dim=-1)
                cl_loss = torch.mean(-torch.log(positive / negative))
                contrastive_loss += cl_loss
                contrastive_count += 1

        for model in models:
            for name, param in model.named_parameters():
                if "weight" in name:
                    cls_loss = cls_loss + param.mean() * 3e-3
        loss = (cls_loss / cls_count) + (contrastive_loss / contrastive_count) * 0.1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {
            "LOSS": loss.item()
        }

    def test(self, data, cache_name, mask=None, output_embedding=False, output_loss=False):
        for model in self.models:
            model.eval()
        with torch.no_grad():
            if output_embedding:
                logits, embedding = self.predict(data, cache_name, mask, output_embedding=True)
            else:
                logits = self.predict(data, cache_name, mask)
            if self.args.clf == 'multi':
                cls_loss = self.bce_loss(logits, data.y.float())
            else:
                cls_loss = self.ce_loss(logits, data.y)
            cls_loss = cls_loss.item()

        labels = data.y if mask is None else data.y[mask]
        if self.args.clf == 'multi':
            preds = (logits > 0).float()
            eval_result = evaluate_multi(preds, labels)
        elif self.args.num_classes == 2:
            preds = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            eval_result = evaluate_single(preds, labels, probs)
        else:
            preds = torch.argmax(logits, dim=-1)
            eval_result = evaluate_single(preds, labels)
        if output_loss:
            return eval_result, cls_loss
        else:
            return eval_result

