import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.readouts import global_readout, grouped_pool_embeddings, grouped_query_readout


class UnifiedPredictionHead(nn.Module):
    def __init__(self, encoder, label_spaces, embedding_dim, edge_pooling="maxmin", graph_pooling="max",
                 temperature=0.5, adapter_bank=None):
        super().__init__()
        self.encoder = encoder
        self.adapter_bank = adapter_bank
        self.embedding_dim = embedding_dim
        self.edge_pooling = edge_pooling
        self.graph_pooling = graph_pooling
        self.temperature = temperature
        self.query_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.label_prototypes = nn.ParameterDict({
            task: nn.Parameter(torch.empty(num_labels, embedding_dim)) for task, num_labels in label_spaces.items()
        })
        for prototype in self.label_prototypes.values():
            nn.init.xavier_uniform_(prototype)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        if self.adapter_bank is not None:
            self.adapter_bank.reset_parameters()
        for module in self.query_mlp:
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        for prototype in self.label_prototypes.values():
            nn.init.xavier_uniform_(prototype)

    def readout(self, node_embeddings, batch):
        graph = batch.h_prime
        if batch.task_type == "hg_cls":
            return global_readout(node_embeddings, graph.batch, self.graph_pooling)

        pooling = "mean" if batch.task_type == "node_cls" else self.edge_pooling
        return grouped_query_readout(node_embeddings, graph.query_mask, graph.batch, graph.num_graphs, pooling)

    def task_key(self, batch):
        return getattr(batch, "task_key", batch.task_type)

    def query_embeddings(self, batch, representation="full"):
        task_key = self.task_key(batch)

        if self.adapter_bank is None:
            if representation != "full":
                raise ValueError("Representation ablation requires task adapters.")
            node_embeddings, _ = self.encoder(batch.h_prime, batch.task_type)
            return self.readout(node_embeddings, batch)

        shared_embeddings, _ = self.encoder(batch.h_prime)
        graph = batch.h_prime

        if batch.task_type == "hg_cls":
            if representation == "shared":
                return global_readout(shared_embeddings, graph.batch, self.graph_pooling)
            residual_embeddings = self.adapter_bank.residual(shared_embeddings, task_key)
            node_embeddings = self.combine_embeddings(shared_embeddings, residual_embeddings, task_key, representation)
            return global_readout(node_embeddings, graph.batch, self.graph_pooling)

        query_mask = graph.query_mask
        query_shared = shared_embeddings[query_mask]
        if representation == "shared":
            query_embeddings = query_shared
        else:
            query_residual = self.adapter_bank.residual(query_shared, task_key)
            query_embeddings = self.combine_embeddings(query_shared, query_residual, task_key, representation)
        pooling = "mean" if batch.task_type == "node_cls" else self.edge_pooling
        return grouped_pool_embeddings(query_embeddings, graph.batch[query_mask], graph.num_graphs, pooling)

    def combine_embeddings(self, shared_embeddings, residual_embeddings, task_key, representation):
        if representation == "shared":
            return shared_embeddings
        if representation == "residual":
            scale = self.adapter_bank.adapters[task_key].residual_scale
            return scale * residual_embeddings
        if representation == "full":
            scale = self.adapter_bank.adapters[task_key].residual_scale
            return shared_embeddings + scale * residual_embeddings
        raise ValueError(f"Unknown representation: {representation}")

    def predict_from_query(self, query_embeddings, batch):
        query_embeddings = F.normalize(query_embeddings + self.query_mlp(query_embeddings), dim=-1)
        prototypes = F.normalize(self.label_prototypes[self.task_key(batch)], dim=-1)
        return query_embeddings @ prototypes.t() / self.temperature

    def predict_from_embeddings(self, node_embeddings, batch):
        return self.predict_from_query(self.readout(node_embeddings, batch), batch)

    def forward(self, batch, representation="full"):
        if self.adapter_bank is None:
            if representation != "full":
                raise ValueError("Representation ablation requires task adapters.")
            node_embeddings, _ = self.encoder(batch.h_prime, batch.task_type)
            return self.predict_from_embeddings(node_embeddings, batch)

        query_embeddings = self.query_embeddings(batch, representation=representation)
        return self.predict_from_query(query_embeddings, batch)
