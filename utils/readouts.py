import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from HNN import MLP


def pool_embeddings(embeddings, pooling, dim=0):
    if pooling == "mean":
        return embeddings.mean(dim=dim)
    if pooling == "max":
        return embeddings.max(dim=dim).values
    if pooling == "maxmin":
        return embeddings.max(dim=dim).values - embeddings.min(dim=dim).values


def grouped_pool_embeddings(embeddings, group_ids, num_groups, pooling):
    group_ids = group_ids.view(-1, 1).expand_as(embeddings)
    shape = (num_groups, embeddings.size(1))

    if pooling == "mean":
        output = embeddings.new_zeros(shape)
        output.scatter_add_(0, group_ids, embeddings)
        counts = torch.bincount(group_ids[:, 0], minlength=num_groups).clamp_min(1).to(embeddings.dtype)
        return output / counts.unsqueeze(-1)

    output = embeddings.new_full(shape, float("-inf"))
    output.scatter_reduce_(0, group_ids, embeddings, reduce="amax", include_self=True)
    if pooling == "max":
        return output

    if pooling == "maxmin":
        minimum = embeddings.new_full(shape, float("inf"))
        minimum.scatter_reduce_(0, group_ids, embeddings, reduce="amin", include_self=True)
        return output - minimum

    raise ValueError(f"Unknown pooling: {pooling}")


def global_readout(node_embeddings, batch, pooling):
    if pooling == "mean":
        return global_mean_pool(node_embeddings, batch)
    num_graphs = int(batch.max().item()) + 1
    return grouped_pool_embeddings(node_embeddings, batch, num_graphs, pooling)


def grouped_query_readout(node_embeddings, query_mask, node_batch, num_graphs, pooling):
    return grouped_pool_embeddings(node_embeddings[query_mask], node_batch[query_mask], num_graphs, pooling)


class MeanAggregator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.classifier = MLP(args.embedding_hidden, args.e_embed_hidden, 1, args.e_embed_layer,
                              args.e_embed_dropout, args.e_embed_norm, InputNorm=False)

    def reset_parameters(self):
        self.classifier.reset_parameters()

    def aggregate(self, embeddings, method="group"):
        dim = 1 if method == "group" else 0
        return pool_embeddings(embeddings, "mean", dim).squeeze()

    def classify(self, embedding):
        return self.classifier(embedding)

    def forward(self, embeddings, method="group"):
        return self.classify(self.aggregate(embeddings, method))


class MaxminAggregator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.classifier = MLP(args.embedding_hidden, args.e_embed_hidden, 1, args.e_embed_layer,
                              args.e_embed_dropout, args.e_embed_norm, InputNorm=False)

    def reset_parameters(self):
        self.classifier.reset_parameters()

    def aggregate(self, embeddings, method="group"):
        dim = 1 if method == "group" else 0
        return pool_embeddings(embeddings, "maxmin", dim)

    def classify(self, embedding):
        return self.classifier(embedding)

    def forward(self, embeddings, method="group"):
        return self.classify(self.aggregate(embeddings, method))


class MaxAggregator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.classifier = MLP(args.embedding_hidden, args.e_embed_hidden, 1, args.e_embed_layer,
                              args.e_embed_dropout, args.e_embed_norm, InputNorm=False)

    def reset_parameters(self):
        self.classifier.reset_parameters()

    def aggregate(self, embeddings, method="group"):
        dim = 1 if method == "group" else 0
        return pool_embeddings(embeddings, "max", dim)

    def classify(self, embedding):
        return self.classifier(embedding)

    def forward(self, embeddings, method="group"):
        return self.classify(self.aggregate(embeddings, method))
