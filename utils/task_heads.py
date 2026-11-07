from collections import defaultdict
import torch
import torch.nn as nn
from HNN import MLP
from .readouts import global_readout


class NodePredictor(nn.Module):
    def __init__(self, encoder, num_targets, args):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.classifier = MLP(args.embedding_hidden, args.n_embed_hidden, num_targets, args.n_embed_layer,
                              args.n_embed_dropout, args.n_embed_norm, InputNorm=False)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, data):
        node_embeddings, _ = self.encoder(data)
        return self.classifier(node_embeddings)


class HyperGPredictor(nn.Module):
    def __init__(self, encoder, num_targets, args):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.pooling = args.pooling
        self.classifier = MLP(args.embedding_hidden, args.g_embed_hidden, num_targets, args.g_embed_layer,
                              args.g_embed_dropout, args.g_embed_norm, InputNorm=False)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, data):
        node_embeddings, _ = self.encoder(data)
        graph_embeddings = global_readout(node_embeddings, data.batch, self.pooling)
        return self.classifier(graph_embeddings)


class EdgePredictor(nn.Module):
    def __init__(self, encoder, aggregator, args):
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.args = args
        self.edge_aggr = args.edge_aggr

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.aggregator.reset_parameters()

    def encoding(self, data):
        return self.encoder(data)

    def aggregate(self, node_embeddings, hyperedges, mode="Train"):
        if self.edge_aggr == "group":
            groups = defaultdict(list)
            for hyperedge in hyperedges:
                groups[len(hyperedge)].append(hyperedge)

            predictions = []
            for group in groups.values():
                features = torch.stack([node_embeddings[hyperedge] for hyperedge in group])
                predictions.append(self.aggregator(features, self.edge_aggr).reshape(-1))
            predictions = torch.cat(predictions)
        else:
            predictions = [self.aggregator(node_embeddings[hyperedge], self.edge_aggr).reshape(-1) for hyperedge in hyperedges]
            predictions = torch.cat(predictions)

        if mode == "Train":
            return predictions
        if mode == "Eval":
            return [prediction.detach() for prediction in predictions]
        raise ValueError(f"Unknown mode: {mode}")
