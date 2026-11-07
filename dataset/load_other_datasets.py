#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as osp
import pickle

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_sparse import coalesce


def load_dataset(path="../hyperGCN/data/", dataset="cora", train_percent=0.025):
    """Load pickle-format single-hypergraph datasets.

    Used by:
    - cora, citeseer, pubmed
    - coauthor_cora, coauthor_dblp, where dataset is mapped to cora/dblp
    """
    with open(osp.join(path, dataset, "features.pickle"), "rb") as f:
        features = pickle.load(f).todense()

    with open(osp.join(path, dataset, "labels.pickle"), "rb") as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)

    print(f"number of nodes:{num_nodes}, feature dimension: {feature_dim}")

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    with open(osp.join(path, dataset, "hypergraph.pickle"), "rb") as f:
        # Dict format: {hyperedge_id: [node_id, ...], ...}
        hypergraph = pickle.load(f)

    print(f"number of hyperedges: {len(hypergraph)}")

    edge_idx = num_nodes
    node_list = []
    edge_list = []

    for he in hypergraph.keys():
        cur_he = hypergraph[he]
        node_list += list(cur_he)
        edge_list += [edge_idx] * len(cur_he)
        edge_idx += 1

    # Build bidirectional star expansion:
    # node -> hyperedge_node and hyperedge_node -> node.
    edge_index = np.array(
        [
            node_list + edge_list,
            edge_list + node_list,
        ],
        dtype=int,
    )
    edge_index = torch.LongTensor(edge_index)

    data = Data(
        x=features,
        edge_index=edge_index,
        y=labels,
    )

    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(
        data.edge_index,
        None,
        total_num_node_id_he_id,
        total_num_node_id_he_id,
    )

    data.n_x = num_nodes
    data.train_percent = train_percent
    data.num_hyperedges = len(hypergraph)

    return data


def load_LE_dataset(path=None, dataset="ModelNet40", train_percent=0.025):
    """Load .content/.edges-format single-hypergraph datasets.

    Used by:
    - 20newsW100
    - ModelNet40
    - NTU2012
    - Mushroom
    - zoo
    """
    print(f"Loading {dataset} dataset...")

    content_path = osp.join(path, dataset, f"{dataset}.content")
    idx_features_labels = np.genfromtxt(content_path, dtype=np.dtype(str))

    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = torch.LongTensor(idx_features_labels[:, -1].astype(float))

    print("load features")

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    edges_path = osp.join(path, dataset, f"{dataset}.edges")
    edges_unordered = np.genfromtxt(edges_path, dtype=np.int32)

    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())),
        dtype=np.int32,
    ).reshape(edges_unordered.shape)

    print("load edges")

    edge_index = edges.T

    assert edge_index[0].max() == edge_index[1].min() - 1
    assert len(np.unique(edge_index)) == edge_index.max() + 1

    num_nodes = edge_index[0].max() + 1
    num_hyperedges = edge_index[1].max() - num_nodes + 1

    # Build bidirectional star expansion.
    edge_index = np.hstack((edge_index, edge_index[::-1, :]))

    data = Data(
        x=torch.FloatTensor(np.array(features[:num_nodes].todense())),
        edge_index=torch.LongTensor(edge_index),
        y=labels[:num_nodes],
    )

    total_num_node_id_he_id = len(np.unique(edge_index))
    data.edge_index, data.edge_attr = coalesce(
        data.edge_index,
        None,
        total_num_node_id_he_id,
        total_num_node_id_he_id,
    )

    data.n_x = num_nodes
    data.y = data.y - data.y.min()
    data.train_percent = train_percent
    data.num_hyperedges = num_hyperedges

    return data