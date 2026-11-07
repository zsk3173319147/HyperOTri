import torch.nn as nn


class ResidualAdapter(nn.Module):
    def __init__(self, embedding_dim, bottleneck_dim=None, dropout=0.0, residual_scale=0.1):
        super().__init__()
        bottleneck_dim = bottleneck_dim or max(1, embedding_dim // 4)
        self.residual_scale = residual_scale
        self.norm = nn.LayerNorm(embedding_dim)
        self.down = nn.Linear(embedding_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.norm.reset_parameters()
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def residual(self, node_embeddings):
        residual = self.norm(node_embeddings)
        residual = self.down(residual)
        residual = self.activation(residual)
        residual = self.dropout(residual)
        return self.up(residual)

    def forward(self, node_embeddings):
        residual = self.residual(node_embeddings)
        return node_embeddings + self.residual_scale * residual


class TaskAdapterBank(nn.Module):
    def __init__(self, embedding_dim, tasks, bottleneck_dim=None, dropout=0.0, residual_scale=0.1):
        super().__init__()
        self.adapters = nn.ModuleDict({
            task: ResidualAdapter(embedding_dim, bottleneck_dim, dropout, residual_scale) for task in tasks
        })

    def reset_parameters(self):
        for adapter in self.adapters.values():
            adapter.reset_parameters()

    def residual(self, node_embeddings, task_type):
        return self.adapters[task_type].residual(node_embeddings)

    def forward(self, node_embeddings, task_type):
        return self.adapters[task_type](node_embeddings)
