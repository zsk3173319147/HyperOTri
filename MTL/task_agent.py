import torch
from torch_geometric.data import Batch, Data
from dataset.data_base import HypergraphData
from .task_batch import TaskBatch


class NodeClsTaskAdapter:
    def __init__(self, builder, labels):
        self.builder = builder
        self.labels = labels
        self.sample_cache = {}

    def build_sample(self, node_id):
        node_id = int(node_id)
        if node_id in self.sample_cache:
            return self.sample_cache[node_id]

        sample = self.builder.build([node_id])
        sample.y = self.labels[node_id].view(1).long().to(sample.x.device)
        self.sample_cache[node_id] = sample
        return sample

    def build_batch(self, node_ids):
        batch = Batch.from_data_list([self.build_sample(node_id) for node_id in node_ids])
        return TaskBatch(batch, "node_cls", batch.y.view(-1).long())


class EdgePredTaskAdapter:
    def __init__(self, builder, exclude_exact_seed_edge=True):
        self.builder = builder
        self.exclude_exact_seed_edge = exclude_exact_seed_edge

    def build_sample(self, candidate_hyperedge, label):
        seed_nodes = tuple(dict.fromkeys(int(node) for node in candidate_hyperedge))
        sample = self.builder.build(seed_nodes, exclude_exact_seed_edge=self.exclude_exact_seed_edge)
        sample.y = torch.as_tensor(label, dtype=torch.float, device=sample.x.device).view(1)
        return sample

    def build_batch(self, candidate_hyperedges, labels):
        if len(candidate_hyperedges) != len(labels):
            raise ValueError("candidate_hyperedges and labels must have equal length")

        samples = [self.build_sample(candidate, label) for candidate, label in zip(candidate_hyperedges, labels)]
        batch = Batch.from_data_list(samples)
        return TaskBatch(batch, "edge_pred", batch.y.view(-1).float())


class GraphClsTaskAdapter:
    def build_batch(self, graphs):
        if isinstance(graphs, Batch):
            batch = graphs.clone()
        else:
            if isinstance(graphs, Data):
                graphs = [graphs]
            batch = Batch.from_data_list([self._ensure_hypergraph_data(graph) for graph in graphs])

        batch.query_mask = torch.ones(batch.num_nodes, dtype=torch.bool, device=batch.x.device)
        batch.x = torch.cat([batch.x, batch.query_mask.to(batch.x.dtype).unsqueeze(-1)], dim=-1)
        return TaskBatch(batch, "hg_cls", batch.y.view(-1).long())

    def _ensure_hypergraph_data(self, graph):
        graph = graph.clone()
        if not isinstance(graph, HypergraphData):
            graph = HypergraphData.from_dict(graph.to_dict())

        edge_ids = graph.hyperedge_index[1]
        if edge_ids.numel() == 0:
            num_hyperedges = 0
        else:
            _, edge_ids = torch.unique(edge_ids, sorted=True, return_inverse=True)
            graph.hyperedge_index = torch.stack([graph.hyperedge_index[0], edge_ids])
            num_hyperedges = int(edge_ids.max().item()) + 1

        graph.num_edges = num_hyperedges
        graph.num_hyperedges = num_hyperedges
        return graph
