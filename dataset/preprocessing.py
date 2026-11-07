import copy
from collections import Counter

import numpy as np
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_scatter import scatter_add


def _as_int(value):
    if torch.is_tensor(value):
        return int(value.view(-1)[0].item())
    return int(value)


def data_processing(args, db):
    data = copy.deepcopy(db.data)

    if args.method in ["AllSetformer", "AllDeepSets"]:
        data = ExtractV2E(data)

        if args.add_self_loop:
            data = Add_Self_Loops(data)

        if args.exclude_self:
            data = expand_edge_index(data)

        data = norm_contruction(data, option=args.normtype)
        db.norm = data.norm.to(args.device)

    else:
        data = ExtractV2E(data)

        if args.add_self_loop:
            data = Add_Self_Loops(data)

        data.edge_index[1] -= data.edge_index[1].min()

    if hasattr(db, "sens") and db.sens is not None:
        db.sens = db.sens.to(args.device)

    db.x = data.x.to(args.device)
    db.hyperedge_index = data.edge_index.to(args.device)
    db.y = data.y.to(args.device)
    db.data = data

    return db


def ExtractV2E(data):
    # Input edge_index is bidirectional star expansion: [V|E ; E|V].
    edge_index = data.edge_index

    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)

    num_nodes = _as_int(data.n_x)
    num_hyperedges = _as_int(data.num_hyperedges)

    if num_nodes + num_hyperedges - 1 != edge_index[0].max().item():
        raise ValueError("num_hyperedges does not match edge_index in ExtractV2E")

    cidx = torch.where(edge_index[0] == num_nodes)[0].min()
    data.edge_index = edge_index[:, :cidx].type(torch.LongTensor)

    return data


def Add_Self_Loops(data):
    # Input edge_index is incidence format: [V ; E].
    edge_index = data.edge_index

    num_nodes = _as_int(data.n_x)
    num_hyperedges = _as_int(data.num_hyperedges)

    if num_nodes + num_hyperedges - 1 != edge_index[1].max().item():
        raise ValueError("num_hyperedges does not match edge_index in Add_Self_Loops")

    hyperedge_appear_freq = Counter(edge_index[1].cpu().numpy())

    skip_node_lst = []
    for edge in hyperedge_appear_freq:
        if hyperedge_appear_freq[edge] == 1:
            skip_node = edge_index[0][torch.where(edge_index[1] == edge)[0].item()]
            skip_node_lst.append(skip_node.item())

    new_edge_idx = edge_index[1].max() + 1
    new_edges = torch.zeros(
        (2, num_nodes - len(set(skip_node_lst))),
        dtype=edge_index.dtype,
    )

    tmp_count = 0
    for i in range(num_nodes):
        if i not in skip_node_lst:
            new_edges[0][tmp_count] = i
            new_edges[1][tmp_count] = new_edge_idx
            new_edge_idx += 1
            tmp_count += 1

    data.totedges = num_hyperedges + num_nodes - len(set(skip_node_lst))

    edge_index = torch.cat((edge_index, new_edges), dim=1)

    _, sorted_idx = torch.sort(edge_index[0])
    data.edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)

    return data


def expand_edge_index(data, edge_th=0):
    edge_index = data.edge_index

    num_nodes = _as_int(data.n_x)
    if hasattr(data, "totedges"):
        num_edges = _as_int(data.totedges)
    else:
        num_edges = _as_int(data.num_hyperedges)

    expanded_n2he_index = []
    cur_he_id = num_nodes

    for he_idx in range(num_nodes, num_edges + num_nodes):
        selected_he = edge_index[:, edge_index[1] == he_idx]
        size_of_he = selected_he.shape[1]

        if edge_th > 0 and size_of_he > edge_th:
            continue

        if size_of_he == 0:
            continue

        if size_of_he == 1:
            new_n2he = selected_he.clone()
            new_n2he[1] = cur_he_id
            expanded_n2he_index.append(new_n2he)
            cur_he_id += 1
            continue

        new_n2he = selected_he.repeat_interleave(size_of_he, dim=1)

        new_edge_ids = torch.LongTensor(
            np.arange(cur_he_id, cur_he_id + size_of_he)
        ).repeat(size_of_he)
        new_n2he[1] = new_edge_ids

        node_id_2_he_id = {}
        for idx in range(size_of_he):
            cur_node_id = selected_he[0][idx].item()
            node_id_2_he_id[cur_node_id] = cur_he_id
            cur_he_id += 1

        keep_mask = torch.BoolTensor([True] * new_n2he.shape[1])
        for col_idx in range(new_n2he.shape[1]):
            node_id = new_n2he[0, col_idx].item()
            edge_id = new_n2he[1, col_idx].item()

            if node_id_2_he_id[node_id] == edge_id:
                keep_mask[col_idx] = False

        new_n2he = new_n2he[:, keep_mask]
        expanded_n2he_index.append(new_n2he)

    if len(expanded_n2he_index) == 0:
        data.edge_index = torch.empty((2, 0), dtype=torch.long)
        return data

    new_edge_index = torch.cat(expanded_n2he_index, dim=1)

    new_order = new_edge_index[0].argsort()
    data.edge_index = new_edge_index[:, new_order].type(torch.LongTensor)

    return data


def norm_contruction(data, option="all_one", TYPE="V2E"):
    if TYPE == "V2E":
        if option == "all_one":
            data.norm = torch.ones_like(data.edge_index[0], dtype=torch.float)

        elif option == "deg_half_sym":
            edge_weight = torch.ones_like(data.edge_index[0], dtype=torch.float)

            cidx = data.edge_index[1].min()

            v_deg = scatter_add(edge_weight, data.edge_index[0], dim=0)
            he_deg = scatter_add(edge_weight, data.edge_index[1] - cidx, dim=0)

            v_norm = v_deg ** (-1 / 2)
            e_norm = he_deg ** (-1 / 2)

            v_norm[torch.isinf(v_norm)] = 0
            e_norm[torch.isinf(e_norm)] = 0

            data.norm = v_norm[data.edge_index[0]] * e_norm[data.edge_index[1] - cidx]

        else:
            raise ValueError(f"Unsupported norm option: {option}")

    elif TYPE == "V2V":
        data.edge_index, data.norm = gcn_norm(
            data.edge_index,
            data.norm,
            add_self_loops=True,
        )

    else:
        raise ValueError(f"Unsupported norm TYPE: {TYPE}")

    return data


def convert_to_hyperedge_index(e_list):
    node_list = []
    edge_list = []

    for edge_idx, cur_he in enumerate(e_list):
        node_list += cur_he
        edge_list += [edge_idx] * len(cur_he)

    edge_index = np.array([node_list, edge_list], dtype=int)
    edge_index = torch.LongTensor(edge_index)

    return edge_index


def compute_degree_list(hyperedge_index, num_nodes=None):
    node_ids = hyperedge_index[0]

    if num_nodes is None:
        num_nodes = node_ids.max().item() + 1

    degrees = torch.bincount(node_ids, minlength=num_nodes)

    return degrees.tolist()