import os.path as osp
import torch
import numpy as np
from torch.utils.data import random_split
from torch_geometric.data import Data

from dataset.preprocessing import convert_to_hyperedge_index, compute_degree_list
from dataset.convert_datasets_to_pygDataset import dataset_Hypergraph


PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
DEFAULT_DATA_ROOT = osp.join(PROJECT_ROOT, "data")


def _as_int(value):
    if torch.is_tensor(value):
        return int(value.view(-1)[0].item())
    return int(value)


class HypergraphData(Data):
    """PyG data with separate node and hyperedge batching offsets."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "hyperedge_index":
            return value.new_tensor(
                [[self.num_nodes], [_as_int(self.num_hyperedges)]]
            )
        return super().__inc__(key, value, *args, **kwargs)


class HyperDataset(object):
    def __init__(self, args):
        self.device = args.device
        self.args = args
        self.name = args.dname
        self.method = args.method

        self.trad_list = [
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

        self.hg_list = [
            "RHG_3",
            "RHG_10",
            "RHG_table",
            "RHG_pyramid",
            "IMDB_dir_form",
            "IMDB_dir_genre",
            "IMDB_wri_form",
            "IMDB_wri_genre",
        ]

        self.data_root = osp.abspath(getattr(args, "data_root", DEFAULT_DATA_ROOT))
        self.trad_path = osp.join(self.data_root, "trad_data")
        self.hg_path = osp.join(self.data_root, "hgcls_data")
        self.pyg_path = osp.join(
            self.data_root,
            "pyg_data",
            "hypergraph_dataset_updated",
        )

        if self.name in self.trad_list:
            self.load_data()
            self._initialization_()
        elif self.name in self.hg_list:
            self.load_multi_data()
        else:
            raise ValueError(f"Undefined dataset: {self.name}")

    # -------------------------- Hypergraph-level Task Dataset Loading ------------------------

    def load_multi_data(self):
        if self.name in ["RHG_3", "RHG_10", "RHG_table", "RHG_pyramid"]:
            folder = "RHG"
        elif self.name in [
            "IMDB_dir_form",
            "IMDB_dir_genre",
            "IMDB_wri_form",
            "IMDB_wri_genre",
        ]:
            folder = "IMDB"
        else:
            raise NotImplementedError(f"Unsupported multi-hypergraph dataset: {self.name}")

        x_list = []

        data_path = osp.join(self.hg_path, folder, f"{self.name}.txt")
        with open(data_path, "r") as f:
            n_g = int(f.readline().strip())

            for _ in range(n_g):
                row = f.readline().strip().split()
                num_v, num_e = int(row[0]), int(row[1])
                g_lbl = [int(x) for x in row[2:]]

                # The original vertex labels are read but not used by DHG-Bench.
                _ = f.readline().strip().split()

                e_list = []
                for _ in range(num_e):
                    row = f.readline().strip().split()
                    e_list.append([int(x) for x in row])

                hyperedge_index = convert_to_hyperedge_index(e_list)
                v_deg = compute_degree_list(hyperedge_index)

                x_list.append(
                    {
                        "num_v": num_v,
                        "num_e": num_e,
                        "v_lbl": v_deg,
                        "g_lbl": g_lbl,
                        "edge_index": hyperedge_index,
                    }
                )

        v_lbl_set, g_lbl_set = set(), set()

        for x in x_list:
            v_lbl_set.update(x["v_lbl"])
            g_lbl_set.update(x["g_lbl"])

        v_lbl_map = {x: i for i, x in enumerate(sorted(v_lbl_set))}
        g_lbl_map = {x: i for i, x in enumerate(sorted(g_lbl_set))}

        ft_dim = len(v_lbl_set)
        n_classes = len(g_lbl_set)

        y_list = []

        for x in x_list:
            x["g_lbl"] = [g_lbl_map[c] for c in x["g_lbl"]]
            x["v_lbl"] = [v_lbl_map[c] for c in x["v_lbl"]]

            x["v_ft"] = np.zeros((x["num_v"], ft_dim))
            row_idx, col_idx = [], []

            for v_idx, v_lbl in enumerate(x["v_lbl"]):
                row_idx.append(v_idx)
                col_idx.append(v_lbl)

            x["v_ft"][row_idx, col_idx] = 1
            y_list.append(x["g_lbl"][0])

        hgs = []

        for i, hg in enumerate(x_list):
            hypergraph = HypergraphData(
                x=torch.FloatTensor(hg["v_ft"]),
                hyperedge_index=hg["edge_index"],
                y=torch.LongTensor([y_list[i]]),
                num_nodes=hg["num_v"],
                num_edges=hg["num_e"],
                num_hyperedges=hg["num_e"],
            )
            hgs.append(hypergraph)

        self.multi_hypergraphs = MultiHypergraphDatasets(
            x=hgs,
            y=torch.LongTensor(y_list),
            num_hypergraphs=n_g,
            num_features=ft_dim,
            num_classes=n_classes,
        )

    # -------------------------- Hypernode/edge-level Task Dataset Loading ------------------------

    def load_data(self):
        dname = self.name

        if dname in ["cora", "citeseer", "pubmed"]:
            p2raw = osp.join(self.trad_path, "cocitation", "")
        elif dname in ["coauthor_cora", "coauthor_dblp"]:
            p2raw = osp.join(self.trad_path, "coauthorship", "")
        elif dname in ["20newsW100", "ModelNet40", "NTU2012", "Mushroom", "zoo"]:
            p2raw = self.trad_path
        else:
            raise ValueError(f"Undefined single-hypergraph dataset: {dname}")

        dataset = dataset_Hypergraph(
            name=dname,
            root=self.pyg_path,
            p2raw=p2raw,
        )

        data = dataset[0]

        if dname == "zoo":
            data.y = data.y - data.y.min()

        if not hasattr(data, "n_x"):
            data.n_x = torch.tensor([data.x.shape[0]])

        if not hasattr(data, "num_hyperedges"):
            data.num_hyperedges = torch.tensor(
                [data.edge_index[0].max() - data.n_x + 1]
            )

        self.data = data
        self.x = data.x
        self.hyperedge_index = data.edge_index
        self.y = data.y

    def _initialization_(self):
        self.num_classes = len(self.y.unique())
        self.num_features = self.x.shape[1]
        self.num_nodes = self.x.shape[0]
        self.num_hyperedges = _as_int(self.data.num_hyperedges)

    def calculate_memory_size(self):
        labels_size = self.y.numel() * self.y.element_size()
        features_size = self.x.numel() * self.x.element_size()
        hyperedge_index_size = self.hyperedge_index.numel() * self.hyperedge_index.element_size()

        total_size_bytes = labels_size + features_size + hyperedge_index_size
        total_size_gb = total_size_bytes / (1024 ** 3)

        print(f"The memory cost of hypergraph: {total_size_bytes} Byte")
        print(f"The memory cost of hypergraph: {total_size_gb:.2f} GB")

    def to(self, device: str):
        self.x = self.x.to(device)
        self.hyperedge_index = self.hyperedge_index.to(device)
        self.y = self.y.to(device)
        return self

    def generate_random_split(
        self,
        train_ratio: float = 0.1,
        val_ratio: float = 0.1,
        seed=None,
    ):
        num_train = int(self.num_nodes * train_ratio)
        num_val = int(self.num_nodes * val_ratio)
        num_test = self.num_nodes - num_train - num_val

        generator = torch.Generator().manual_seed(seed) if seed is not None else torch.default_generator

        train_set, val_set, test_set = random_split(
            torch.arange(0, self.num_nodes),
            (num_train, num_val, num_test),
            generator=generator,
        )

        train_mask = torch.zeros((self.num_nodes,), device=self.device, dtype=torch.bool)
        val_mask = torch.zeros((self.num_nodes,), device=self.device, dtype=torch.bool)
        test_mask = torch.zeros((self.num_nodes,), device=self.device, dtype=torch.bool)

        train_mask[train_set.indices] = True
        val_mask[val_set.indices] = True
        test_mask[test_set.indices] = True

        return {
            "train": train_mask,
            "valid": val_mask,
            "test": test_mask,
        }


class MultiHypergraphDatasets:
    def __init__(self, x, y, num_hypergraphs, num_features, num_classes):
        self.x = x
        self.y = y
        self.num_hypergraphs = num_hypergraphs
        self.num_features = num_features
        self.num_classes = num_classes

    def __len__(self):
        return self.num_hypergraphs

    def __getitem__(self, idx):
        return self.x[idx]
