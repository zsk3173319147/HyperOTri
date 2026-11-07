from collections import defaultdict
import torch
from dataset.data_base import HypergraphData


class PropagationSubgraphBuilder:
    def __init__(self, x, hyperedge_index, context_hops=1, max_hyperedges=8):
        self.x = x
        self.context_hops = int(context_hops)
        self.max_hyperedges = int(max_hyperedges)
        self.node_edges = [set() for _ in range(x.size(0))]
        edge_nodes = defaultdict(set)

        incidence = hyperedge_index.detach().cpu()
        for node_id, edge_id in zip(incidence[0].tolist(), incidence[1].tolist()):
            self.node_edges[node_id].add(edge_id)
            edge_nodes[edge_id].add(node_id)

        self.node_edges = [tuple(sorted(edges)) for edges in self.node_edges]
        self.edge_nodes = {edge_id: tuple(sorted(nodes)) for edge_id, nodes in edge_nodes.items()}
        self.edge_node_sets = {edge_id: frozenset(nodes) for edge_id, nodes in self.edge_nodes.items()}
        self.edge_ids_by_nodes = defaultdict(set)
        for edge_id, nodes in self.edge_node_sets.items():
            self.edge_ids_by_nodes[nodes].add(edge_id)

    def build(self, seed_nodes, exclude_exact_seed_edge=False, excluded_hyperedge_ids=()):
        seeds = list(dict.fromkeys(int(node) for node in seed_nodes))
        seed_set = set(seeds)
        blocked_edges = {int(edge_id) for edge_id in excluded_hyperedge_ids}
        if exclude_exact_seed_edge:
            blocked_edges.update(self.edge_ids_by_nodes.get(frozenset(seed_set), ()))

        selected_nodes = set(seed_set)
        selected_edges = []
        frontier = set(seed_set)

        for _ in range(self.context_hops):
            remaining = self.max_hyperedges - len(selected_edges)
            if remaining <= 0 or not frontier:
                break

            candidates = set()
            for node_id in frontier:
                for edge_id in self.node_edges[node_id]:
                    if edge_id not in blocked_edges:
                        candidates.add(edge_id)
            if not candidates:
                break

            candidates = sorted(candidates, key=lambda edge_id: (
                -len(self.edge_node_sets[edge_id] & frontier),
                -len(self.edge_node_sets[edge_id] & seed_set),
                edge_id,
            ))
            chosen = candidates[:remaining]
            selected_edges.extend(chosen)
            blocked_edges.update(chosen)

            expanded_nodes = set()
            for edge_id in chosen:
                expanded_nodes.update(self.edge_nodes[edge_id])
            frontier = expanded_nodes - selected_nodes
            selected_nodes.update(expanded_nodes)

        global_node_ids = seeds + sorted(selected_nodes - seed_set)
        node_to_local = {node_id: local_id for local_id, node_id in enumerate(global_node_ids)}
        incidence_nodes, incidence_edges = [], []

        for local_edge_id, global_edge_id in enumerate(selected_edges):
            nodes = self.edge_nodes[global_edge_id]
            incidence_nodes.extend(node_to_local[node_id] for node_id in nodes)
            incidence_edges.extend([local_edge_id] * len(nodes))

        global_hyperedge_ids = list(selected_edges)
        covered_nodes = set(incidence_nodes)
        for node_id in range(len(global_node_ids)):
            if node_id not in covered_nodes:
                incidence_nodes.append(node_id)
                incidence_edges.append(len(global_hyperedge_ids))
                global_hyperedge_ids.append(-1)

        device = self.x.device
        global_node_ids = torch.tensor(global_node_ids, dtype=torch.long, device=device)
        query_mask = torch.arange(global_node_ids.numel(), device=device) < len(seeds)
        hyperedge_index = torch.tensor([incidence_nodes, incidence_edges], dtype=torch.long, device=device)
        global_hyperedge_ids = torch.tensor(global_hyperedge_ids, dtype=torch.long, device=device)
        x = self.x[global_node_ids]
        x = torch.cat([x, query_mask.to(x.dtype).unsqueeze(-1)], dim=-1)

        return HypergraphData(
            x=x, hyperedge_index=hyperedge_index, query_mask=query_mask,
            global_node_ids=global_node_ids, global_hyperedge_ids=global_hyperedge_ids,
            num_nodes=global_node_ids.numel(), num_edges=global_hyperedge_ids.numel(),
            num_hyperedges=global_hyperedge_ids.numel()
        )
