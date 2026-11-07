#!/usr/bin/env python
# coding: utf-8

import os
import os.path as osp
import pickle

import torch
from torch_geometric.data import InMemoryDataset

from dataset.load_other_datasets import load_dataset, load_LE_dataset


PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
DEFAULT_DATA_ROOT = osp.join(PROJECT_ROOT, "data")
DEFAULT_PYG_ROOT = osp.join(
    DEFAULT_DATA_ROOT,
    "pyg_data",
    "hypergraph_dataset_updated",
)


def _default_raw_root(name):
    if name in ["cora", "citeseer", "pubmed"]:
        return osp.join(DEFAULT_DATA_ROOT, "trad_data", "cocitation")
    if name in ["coauthor_cora", "coauthor_dblp"]:
        return osp.join(DEFAULT_DATA_ROOT, "trad_data", "coauthorship")
    if name in ["20newsW100", "ModelNet40", "NTU2012", "Mushroom", "zoo"]:
        return osp.join(DEFAULT_DATA_ROOT, "trad_data")
    return None


def _resolve_existing_dir(path):
    path = osp.expanduser(path)
    candidates = [path] if osp.isabs(path) else [osp.abspath(path), osp.join(PROJECT_ROOT, path)]

    for candidate in candidates:
        if osp.isdir(candidate):
            return osp.abspath(candidate)

    return None


class dataset_Hypergraph(InMemoryDataset):
    """Convert selected single-hypergraph datasets to cached PyG Data."""

    def __init__(
        self,
        root=None,
        name=None,
        p2raw=None,
        train_percent=0.01,
        transform=None,
        pre_transform=None,
    ):
        existing_dataset = [
            "cora",
            "citeseer",
            "pubmed",
            "coauthor_cora",
            "coauthor_dblp",
            "20newsW100",
            "ModelNet40",
            "NTU2012",
            "Mushroom",
            "zoo",
        ]

        if name not in existing_dataset:
            raise ValueError(f"name of hypergraph dataset must be one of: {existing_dataset}")

        self.name = name
        self._train_percent = train_percent
        root = DEFAULT_PYG_ROOT if root is None else osp.abspath(root)
        p2raw = _default_raw_root(name) if p2raw is None else p2raw

        resolved_p2raw = _resolve_existing_dir(p2raw)
        if resolved_p2raw is None:
            raise ValueError(f'path to raw hypergraph dataset "{p2raw}" does not exist!')
        self.p2raw = resolved_p2raw

        os.makedirs(root, exist_ok=True)

        self.root = root
        self.myraw_dir = osp.join(root, self.name, "raw")
        self.myprocessed_dir = osp.join(root, self.name, "processed")

        super(dataset_Hypergraph, self).__init__(
            osp.join(root, name),
            transform,
            pre_transform,
        )

        self._data, self.slices = torch.load(
            self.processed_paths[0],
            weights_only=False,
        )
        self.train_percent = self._data.train_percent

    @property
    def raw_file_names(self):
        return [self.name]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    @property
    def num_features(self):
        return self._data.num_node_features

    def download(self):
        p2f = osp.join(self.myraw_dir, self.raw_file_names[0])
        if osp.isfile(p2f):
            return

        if self.name in ["cora", "citeseer", "pubmed"]:
            tmp_data = load_dataset(
                path=self.p2raw,
                dataset=self.name,
                train_percent=self._train_percent,
            )

        elif self.name in ["coauthor_cora", "coauthor_dblp"]:
            dataset_name = self.name.split("_")[-1]
            tmp_data = load_dataset(
                path=self.p2raw,
                dataset=dataset_name,
                train_percent=self._train_percent,
            )

        elif self.name in ["20newsW100", "ModelNet40", "NTU2012", "Mushroom", "zoo"]:
            tmp_data = load_LE_dataset(
                path=self.p2raw,
                dataset=self.name,
                train_percent=self._train_percent,
            )

        else:
            raise ValueError(f"Undefined dataset: {self.name}")

        save_data_to_pickle(
            tmp_data,
            p2root=self.myraw_dir,
            file_name=self.raw_file_names[0],
        )

    def process(self):
        p2f = osp.join(self.myraw_dir, self.raw_file_names[0])
        with open(p2f, "rb") as f:
            data = pickle.load(f)

        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f"{self.name}()"


def save_data_to_pickle(data, p2root=None, file_name=None):
    p2root = DEFAULT_DATA_ROOT if p2root is None else p2root
    tmp_data_name = file_name if file_name is not None else "Hypergraph_star_expansion_dataset"
    p2he_star_expansion = osp.join(p2root, tmp_data_name)

    os.makedirs(p2root, exist_ok=True)

    with open(p2he_star_expansion, "bw") as f:
        pickle.dump(data, f)

    return p2he_star_expansion
