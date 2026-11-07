import math
import time
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from HNN.preprocessing import algo_preprocessing
from utils.metrics import edge_evaluation_printer, evaluate, evaluate_edge, evaluate_hypegraph, hg_evaluation_printer
from utils.utils import add_self_loop_hyperedges, relabel_hyperedge_index


def prepare_hypergraph_batch(model, batch, args):
    encoder = getattr(model, "encoder", model)
    if hasattr(encoder, "structure"):
        encoder.structure = None

    batch.hyperedge_index, _ = relabel_hyperedge_index(batch.hyperedge_index)
    if args.method == "HyperGCN":
        batch.hyperedge_index = add_self_loop_hyperedges(batch.hyperedge_index, batch.num_nodes)

    batch = algo_preprocessing(batch, args)
    if args.method in ["AllSetformer", "AllDeepSets"]:
        batch.norm = torch.ones(batch.hyperedge_index.size(1), dtype=batch.x.dtype, device=batch.hyperedge_index.device)
    return batch.to(args.device)


def balanced_train_weights(train_loader, num_classes, device):
    labels = [graph.y.view(-1).cpu() for graph in train_loader.dataset]
    if len(labels) == 0:
        raise ValueError("cannot compute class weights from an empty training set")

    labels = torch.cat(labels).long()
    counts = torch.bincount(labels, minlength=num_classes).float()
    present = counts > 0
    weights = torch.zeros(num_classes)
    weights[present] = labels.numel() / (present.sum() * counts[present])
    return weights.to(device)


class Trainer:
    def __init__(self, args, **kwargs):
        self.args = args
        self.device = args.device
        self.train_time = 0.0

    def semi_node_cls_training(self, model, data, masks, args):
        model = model.to(args.device)
        model.reset_parameters()
        data = data.to(args.device)
        masks = {name: torch.as_tensor(mask, device=args.device) for name, mask in masks.items()}

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        start_time = time.time()

        for epoch in trange(args.epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits[masks["train"]], data.y[masks["train"]])
            loss.backward()
            optimizer.step()

            if (epoch + 1) % args.display_step == 0 or epoch + 1 == args.epochs:
                result = evaluate(model, data, masks)
                print(f"Epoch: {epoch + 1:02d}, Training loss: {loss.item():.4f}, Train Acc: {100 * result[0]:.2f}%, Valid Acc: {100 * result[1]:.2f}%, Test Acc: {100 * result[2]:.2f}%")

        self.train_time = time.time() - start_time
        print(f"Training Time: {self.train_time:.2f}")
        return model

    def semi_edge_pred_training(self, model, data, batch_loaders, args):
        model = model.to(args.device)
        model.reset_parameters()
        data = data.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        criterion = nn.BCEWithLogitsLoss()

        pos_loader = batch_loaders["train_pos"]
        neg_loader = batch_loaders["train_neg"]
        pos_batches = math.ceil(len(pos_loader.hyperedges) / pos_loader.batch_size)
        neg_batches = math.ceil(len(neg_loader.hyperedges) / neg_loader.batch_size)
        num_batches = max(pos_batches, neg_batches)
        start_time = time.time()

        for epoch in tqdm(range(args.epochs)):
            model.train()
            pos_loader.train()
            neg_loader.train()
            pos_loader.start_epoch()
            neg_loader.start_epoch()
            total_loss = 0.0

            for _ in range(num_batches):
                optimizer.zero_grad()
                node_embeddings, _ = model.encoding(data)
                pos_edges, pos_labels, _ = pos_loader.next()
                neg_edges, neg_labels, _ = neg_loader.next()
                pos_logits = model.aggregate(node_embeddings, pos_edges, mode="Train").reshape(-1)
                neg_logits = model.aggregate(node_embeddings, neg_edges, mode="Train").reshape(-1)
                loss = criterion(pos_logits, pos_labels.view(-1)) + criterion(neg_logits, neg_labels.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % args.display_step == 0 or epoch + 1 == args.epochs:
                train_metrics, val_metrics, test_metrics = evaluate_edge(model, data, batch_loaders)
                print(f"Epoch: {epoch + 1:02d}, Training loss: {total_loss / num_batches:.4f}")
                edge_evaluation_printer(train_metrics, val_metrics, test_metrics)

        self.train_time = time.time() - start_time
        print(f"Training Time: {self.train_time:.2f}")
        return model

    def semi_hg_cls_training(self, model, data, batch_loaders, args):
        model = model.to(args.device)
        model.reset_parameters()
        train_loader = batch_loaders["train"]
        weights = balanced_train_weights(train_loader, int(data.num_classes), args.device) if getattr(args, "use_weighted_loss", False) else None
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        start_time = time.time()

        for epoch in tqdm(range(args.epochs)):
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                batch = prepare_hypergraph_batch(model, batch, args)
                optimizer.zero_grad()
                loss = criterion(model(batch), batch.y.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % args.display_step == 0 or epoch + 1 == args.epochs:
                result = evaluate_hypegraph(model, batch_loaders, args)
                print(f"Epoch: {epoch + 1:02d}, Training loss: {total_loss / len(train_loader):.4f}")
                hg_evaluation_printer(result)

        self.train_time = time.time() - start_time
        print(f"Training Time: {self.train_time:.2f}")
        return model

    def training(self, model, data, args, seed_split=None, task_type="node_cls"):
        if task_type == "node_cls":
            return self.semi_node_cls_training(model, data, seed_split, args)
        if task_type == "edge_pred":
            return self.semi_edge_pred_training(model, data, seed_split, args)
        if task_type == "hg_cls":
            return self.semi_hg_cls_training(model, data, seed_split, args)

