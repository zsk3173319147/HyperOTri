import os
import random
from collections import defaultdict

import numpy as np
import torch

from dataset.edge_sampler import neg_generator
from utils.utils import fix_seed

EDGE_TRAND_PROTOCOL = "ahp_trand_v4_semantic_edges_matched_sizes"


def edge_split_is_current(data_dict):
    return data_dict.get("split_protocol") == EDGE_TRAND_PROTOCOL


def observed_hyperedges_for_embedding(data_dict):
    observed = data_dict.get("ground_train", [])
    if observed:
        return observed
    return data_dict.get("train_only_pos", [])


def split_positive_hyperedges(data_dict, split):
    if split in ["context", "observed", "embedding"]:
        return observed_hyperedges_for_embedding(data_dict)
    if split == "train":
        train_pos = data_dict.get("train_only_pos", [])
        if train_pos:
            return train_pos
        return data_dict.get("ground_train", [])
    if split in ["val", "valid"]:
        return data_dict["ground_valid"] + data_dict["valid_only_pos"]
    if split == "test":
        return data_dict["test_pos"]
    raise ValueError(f"Unknown edge prediction split: {split}")

def build_hyperedge_index_from_hyperedges(hyperedges, device=None):
    rows = []
    cols = []
    for edge_id, hyperedge in enumerate(hyperedges):
        nodes = sorted(set(int(node) for node in hyperedge))
        rows.extend(nodes)
        cols.extend([edge_id] * len(nodes))

    if not rows:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    return torch.tensor([rows, cols], dtype=torch.long, device=device)

def generate_split_hyperedges(data,args,seed):

    hyperedge_index = data.hyperedge_index.to('cpu')

    hyperedges = defaultdict(set)

    for node, edge in zip(*hyperedge_index.tolist()):
        hyperedges[edge].add(node)

    # Hyperedges with different raw IDs may contain exactly the same node set.
    # Split semantic hyperedges only once to prevent cross-split leakage.
    HE = list(dict.fromkeys(frozenset(nodes) for nodes in hyperedges.values()))
    
    base_cover = get_cover_idx(HE)
    union = get_union(HE)
    tmp = [HE[idx] for idx in base_cover]
    assert union == get_union(tmp)
    base_num = len(base_cover)
    
    split_dir = os.path.join(args.edge_save_dir, args.edge_split_mode, args.dname)
    os.makedirs(split_dir, exist_ok=True)  
    
    seed_base = 42  
    fix_seed(seed_base+seed) 
    
    # ground 60%, train 10(+50)%, validation 10(+10)%, test 20%
    # ground_num = int(0.6*len(HE)) - base_num
    ground_num = max(int(0.6*len(HE)) - base_num,0)
    total_idx = list(range(len(HE))) 
    ground_idx = list(set(total_idx)-set(base_cover))
    ground_idx = random.sample(ground_idx, ground_num)      
    ground_num += base_num
    ground_idx += base_cover
    ground_valid_num = ground_num//6
    ground_valid_idx = random.sample(ground_idx, ground_valid_num)
    ground_train_num = ground_num - ground_valid_num
    
    ground_train_data = []
    ground_valid_data = []
    pred_data = []
    for idx in total_idx :
        if idx in ground_idx:
            if idx in ground_valid_idx:
                ground_valid_data.append(HE[idx])
            else:
                ground_train_data.append(HE[idx])
        else :
            pred_data.append(HE[idx])
            
    valid_only_num = int(0.25*len(pred_data))
    train_only_num = int(0.25*len(pred_data))
    test_num = len(pred_data) - (valid_only_num + train_only_num)
    
    random.shuffle(pred_data)
    train_only_data = pred_data[:train_only_num]
    valid_only_data = pred_data[train_only_num:-test_num]
    test_data = pred_data[-test_num:]
    
    # negative sampling
    GP_train = ground_train_data + train_only_data
    GP_valid = ground_valid_data + ground_train_data + train_only_data + valid_only_data
    GP_test = GP_valid
    all_positive_hyperedges = set(HE)

    train_mns, train_sns, train_cns = neg_generator(GP_train, len(train_only_data), forbidden_HE=all_positive_hyperedges)
    valid_mns, valid_sns, valid_cns = neg_generator(GP_valid, ground_valid_num+valid_only_num, forbidden_HE=all_positive_hyperedges)
    test_mns, test_sns, test_cns = neg_generator(GP_test, test_num, forbidden_HE=all_positive_hyperedges)

    expected_negative_counts = {
        "train_mns": (train_mns, len(train_only_data)),
        "train_sns": (train_sns, len(train_only_data)),
        "train_cns": (train_cns, len(train_only_data)),
        "valid_mns": (valid_mns, ground_valid_num + valid_only_num),
        "valid_sns": (valid_sns, ground_valid_num + valid_only_num),
        "valid_cns": (valid_cns, ground_valid_num + valid_only_num),
        "test_mns": (test_mns, test_num),
        "test_sns": (test_sns, test_num),
        "test_cns": (test_cns, test_num),
    }
    for name, (samples, expected) in expected_negative_counts.items():
        if len(samples) != expected:
            raise RuntimeError(
                f"{name} generated {len(samples)} negatives, expected {expected}."
            )
    
    # positive samples
    ground_train_data = [list(edge) for edge in ground_train_data]
    ground_valid_data = [list(edge) for edge in ground_valid_data]
    train_only_data = [list(edge) for edge in train_only_data]
    valid_only_data = [list(edge) for edge in valid_only_data]
    test_data = [list(edge) for edge in test_data]
    
    print(f"context edges {len(ground_train_data)}")
    print(f"train pos {len(train_only_data)}, neg {len(train_mns)}")
    print(f"valid pos {len(ground_valid_data)} + {len(valid_only_data)} = {len(ground_valid_data + valid_only_data)}, neg {len(valid_mns)}")
    print(f"test pos {len(test_data)}, neg {len(test_mns)}")
    
    split_path = os.path.join(split_dir, f"split_{seed}.pt")
    torch.save({'split_protocol': EDGE_TRAND_PROTOCOL, 'ground_train': ground_train_data, 'ground_valid': ground_valid_data, \
        'train_only_pos': train_only_data, 'train_mns': train_mns, 'train_sns' : train_sns, 'train_cns' : train_cns,\
        'valid_only_pos': valid_only_data, 'valid_mns': valid_mns, 'valid_sns' : valid_sns, 'valid_cns' : valid_cns, \
        'test_pos': test_data, 'test_mns': test_mns, 'test_sns' : test_sns, 'test_cns' : test_cns},
        split_path)

def generate_edge_loaders(data_dict, args):

    device = args.device
    
    train_pos_loader = load_train(data_dict, args.edge_batch_size, device,label="pos") # only positives
    train_neg_loader = load_train(data_dict, args.edge_batch_size, device,label=args.ns_method) # only positives

    val_pos_loader = load_val(data_dict, args.edge_batch_size, device, label="pos")
    val_neg_sns_loader = load_val(data_dict, args.edge_batch_size, device, label="sns")
    val_neg_mns_loader = load_val(data_dict, args.edge_batch_size, device, label="mns")
    val_neg_cns_loader = load_val(data_dict, args.edge_batch_size, device, label="cns")

    test_pos_loader = load_test(data_dict, args.edge_batch_size, device, label="pos")
    test_neg_sns_loader = load_test(data_dict, args.edge_batch_size, device, label="sns")
    test_neg_mns_loader = load_test(data_dict, args.edge_batch_size, device, label="mns")
    test_neg_cns_loader = load_test(data_dict, args.edge_batch_size, device, label="cns")

    batch_loaders = {
        'train_pos': train_pos_loader,
        'train_neg': train_neg_loader,
        'val_pos': val_pos_loader,
        'val_neg_sns':val_neg_sns_loader,
        'val_neg_mns':val_neg_mns_loader,
        'val_neg_cns':val_neg_cns_loader,
        'test_pos': test_pos_loader,
        'test_neg_sns':test_neg_sns_loader,
        'test_neg_mns':test_neg_mns_loader,
        'test_neg_cns':test_neg_cns_loader,
    }
    
    return batch_loaders

def load_train(data_dict, bs, device, label):
    if label=="pos":
        train_pos = split_positive_hyperedges(data_dict, "train")
        train_pos_label = [1 for i in range(len(train_pos))]
        train_batchloader = HEBatchGenerator(
            train_pos,
            train_pos_label,
            bs,
            device,
            test_generator=False,
            loader_name="train_pos",
        )
    elif label =="mixed":
        train_pos = split_positive_hyperedges(data_dict, "train")
        train_neg = _mixed_training_negative_pool(data_dict)
        train_neg_label = [0 for i in range(len(train_neg))]
        train_batchloader = HEBatchGenerator(
            train_neg,
            train_neg_label,
            bs,
            device,
            test_generator=False,
            loader_name="train_neg_mixed",
            epoch_size=len(train_pos),
        )
    else:
        train_neg = data_dict[f"train_{label}"]
        train_neg_label = [0 for i in range(len(train_neg))]
        train_batchloader = HEBatchGenerator(
            train_neg,
            train_neg_label,
            bs,
            device,
            test_generator=False,
            loader_name=f"train_{label}",
        )
    
    return train_batchloader


def _mixed_training_negative_pool(data_dict):
    mixed = []
    seen = set()
    for name in ("train_sns", "train_mns", "train_cns"):
        for hyperedge in data_dict[name]:
            key = frozenset(hyperedge)
            if key in seen:
                continue
            mixed.append(hyperedge)
            seen.add(key)
    return mixed

def load_val(data_dict, bs, device, label):
    if label=="pos":
        val = split_positive_hyperedges(data_dict, "val")
        val_label = [1 for i in range(len(val))]
    else:
        val = data_dict[f"valid_{label}"]
        val_label = [0 for i in range(len(val))]
    loader_name = "val_pos" if label == "pos" else f"val_neg_{label}"
    val_batchloader = HEBatchGenerator(
        val,
        val_label,
        bs,
        device,
        test_generator=True,
        loader_name=loader_name,
    )
    return val_batchloader

def load_test(data_dict, bs, device, label):
    if label=="pos":
        test = split_positive_hyperedges(data_dict, "test")
        test_label = [1 for i in range(len(test))]
    else:
        test = data_dict[f"test_{label}"]
        test_label = [0 for i in range(len(test))]
    loader_name = "test_pos" if label == "pos" else f"test_neg_{label}"
    test_batchloader = HEBatchGenerator(
        test,
        test_label,
        bs,
        device,
        test_generator=True,
        loader_name=loader_name,
    )
    return test_batchloader

# Batch Generator
class HEBatchGenerator(object):
    def __init__(
        self,
        hyperedges,
        labels,
        batch_size,
        device,
        test_generator=False,
        loader_name="hyperedge_loader",
        epoch_size=None,
    ):
        """Creates an instance of HyperedgeGroupBatchGenerator.
        
        Args:
            hyperedges: List(frozenset). List of hyperedges.
            labels: list. Labels of hyperedges.
            batch_size: int. Maximum samples per batch. Values larger than
                the dataset size use all available samples in one batch.
            test_generator: bool. Whether batch generator is test generator.
            loader_name: str. Name used in validation errors.
            epoch_size: Number of pool samples exposed in each training epoch.
        """
        if batch_size <= 0:
            raise ValueError(f"{loader_name} requires edge_batch_size > 0, got {batch_size}.")

        num_samples = len(hyperedges)
        if num_samples == 0:
            raise ValueError(
                f"{loader_name} has no samples. Check edge split generation or negative sampling."
            )

        if epoch_size is None:
            epoch_size = num_samples
        if epoch_size <= 0 or epoch_size > num_samples:
            raise ValueError(
                f"{loader_name} requires 0 < epoch_size <= {num_samples}, "
                f"got {epoch_size}."
            )

        self.requested_batch_size = batch_size
        self.epoch_size = epoch_size
        self.batch_size = min(batch_size, epoch_size)
        self._pool_hyperedges = list(hyperedges)
        self._pool_labels = list(labels)
        self.hyperedges = []
        self.labels = []
        self._cursor = 0
        self.device = device
        self.test_generator = test_generator
        self.loader_name = loader_name
        self.start_epoch()
    
    def eval(self):
        self.test_generator = True

    def train(self):
        self.test_generator = False

    def start_epoch(self):
        idcs = np.random.permutation(len(self._pool_hyperedges))[:self.epoch_size]
        self.hyperedges = [self._pool_hyperedges[i] for i in idcs]
        self.labels = [self._pool_labels[i] for i in idcs]
        self._cursor = 0

    def shuffle(self):
        idcs = np.arange(len(self.hyperedges))
        np.random.shuffle(idcs)
        self.hyperedges = [self.hyperedges[i] for i in idcs]
        self.labels = [self.labels[i] for i in idcs]
  
    def __iter__(self):
        self._cursor = 0
        return self
    
    def next(self):
        return self.__next__()

    def __next__(self):
        if self.test_generator:
            return self.next_test_batch()
        else:
            return self.next_train_batch()

    def next_train_batch(self):
        end = min(self._cursor + self.batch_size, len(self.hyperedges))
        hyperedges = self.hyperedges[self._cursor:end]
        labels = self.labels[self._cursor:end]
        is_last = end == len(self.hyperedges)

        self._cursor = 0 if is_last else end
        hyperedges = [torch.LongTensor(edge).to(self.device) for edge in hyperedges]
        labels = torch.FloatTensor(labels).to(self.device)

        if is_last:
            self.shuffle()
        return hyperedges, labels, is_last

    def next_test_batch(self):
        ncursor = self._cursor+self.batch_size
        if ncursor >= len(self.hyperedges):
            hyperedges = self.hyperedges[self._cursor:]
            labels = self.labels[self._cursor:]
            self._cursor = 0
            hyperedges = [torch.LongTensor(edge).to(self.device) for edge in hyperedges]
            labels = torch.FloatTensor(labels).to(self.device)
            
            return hyperedges, labels, True
        
        hyperedges = self.hyperedges[
            self._cursor:self._cursor + self.batch_size]
        
        labels = self.labels[
            self._cursor:self._cursor + self.batch_size]

        hyperedges = [torch.LongTensor(edge).to(self.device) for edge in hyperedges]
        labels = torch.FloatTensor(labels).to(self.device)
       
        self._cursor = ncursor % len(self.hyperedges)
        return hyperedges, labels, False

def get_union(union):
    ind = []    
    for s in union :
        ind+=list(s)
    return set(ind)

def set_cover(universe, subsets):
    elements = set(e for s in subsets for e in s)
    if elements != universe:
        return None, None
    covered = set()
    cover = []
    idx = []
    while covered != elements:
        subset = max(subsets, key=lambda s: len(s - covered))
        cover.append(subset)
        idx.append(subsets.index(subset))
        covered |= subset
    return cover, idx

def get_cover_idx(HE):
    universe = get_union(HE)
    tmp_HE = [set(edge) for edge in HE]
    _, cover_idx = set_cover(universe, tmp_HE)
    return cover_idx
