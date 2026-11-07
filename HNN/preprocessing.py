import copy
from collections import defaultdict

import numpy as np
import torch
from torch_scatter import scatter_add


def algo_preprocessing(data, args):
    if args.method == "HNHN":
        data = generate_HNHN_norm(data, args)

    return data


def generate_HNHN_norm(data, args):
    edge_index = copy.deepcopy(data.hyperedge_index).to("cpu")

    ones = torch.ones(edge_index.shape[1], device=edge_index.device)

    alpha = args.HNHN_alpha
    beta = args.HNHN_beta
    device = getattr(args, "device", data.x.device)

    dv = scatter_add(ones, edge_index[0], dim=0)
    de = scatter_add(ones, edge_index[1], dim=0)

    d_e_alpha = de ** alpha
    d_e_alpha[d_e_alpha == float("inf")] = 0

    d_v_alpha = scatter_add(de[edge_index[1]], edge_index[0], dim=0)

    d_v_beta = dv ** beta
    d_v_beta[d_v_beta == float("inf")] = 0

    d_e_beta = scatter_add(dv[edge_index[0]], edge_index[1], dim=0)

    d_v_alpha_inv = 1.0 / d_v_alpha
    d_v_alpha_inv[d_v_alpha_inv == float("inf")] = 0

    d_e_beta_inv = 1.0 / d_e_beta
    d_e_beta_inv[d_e_beta_inv == float("inf")] = 0

    data.D_e_alpha = d_e_alpha.float().to(device)
    data.D_v_alpha_inv = d_v_alpha_inv.float().to(device)
    data.D_v_beta = d_v_beta.float().to(device)
    data.D_e_beta_inv = d_e_beta_inv.float().to(device)

    return data


def get_HyperGCN_He_dict(data):
    edge_index = np.array(data.hyperedge_index.detach().cpu())
    he_dict = defaultdict(list)

    for node, hyperedge in zip(edge_index[0, :], edge_index[1, :]):
        he_dict[int(hyperedge)].append(int(node))

    return dict(he_dict)


def update(Se, Ie, mediator, weights, c):
    if (Se, mediator) not in weights:
        weights[(Se, mediator)] = 0
    weights[(Se, mediator)] += float(1 / c)

    if (Ie, mediator) not in weights:
        weights[(Ie, mediator)] = 0
    weights[(Ie, mediator)] += float(1 / c)

    if (mediator, Se) not in weights:
        weights[(mediator, Se)] = 0
    weights[(mediator, Se)] += float(1 / c)

    if (mediator, Ie) not in weights:
        weights[(mediator, Ie)] = 0
    weights[(mediator, Ie)] += float(1 / c)

    return weights