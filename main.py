import os.path as osp
from copy import copy

import torch

from HNN import HGNN, HNHN, HyperGCN, SetGNN, UniGNN
from HNN.preprocessing import algo_preprocessing
from dataset.data_base import HyperDataset
from dataset.edge_loaders import (
    EDGE_TRAND_PROTOCOL,
    build_hyperedge_index_from_hyperedges,
    edge_split_is_current,
    generate_edge_loaders,
    generate_split_hyperedges,
    observed_hyperedges_for_embedding,
)
from dataset.hg_loaders import generate_hg_loaders, generate_split_hypergraphs
from dataset.preprocessing import data_processing, norm_contruction
from parameter_parser_normal import parameter_parser
from utils.eval_agent import Evaluator
from utils.readouts import MaxAggregator, MaxminAggregator, MeanAggregator
from utils.task_heads import EdgePredictor, HyperGPredictor, NodePredictor
from utils.train_agent import Trainer
from utils.utils import add_self_loop_hyperedges, fix_seed


ENCODERS = {
    "HGNN": HGNN,
    "HNHN": HNHN,
    "HyperGCN": HyperGCN,
    "UniGIN": UniGNN,
}

EDGE_AGGREGATORS = {
    "mean": MeanAggregator,
    "max": MaxAggregator,
    "maxmin": MaxminAggregator,
}


def resolve_device(device_name):
    requested = torch.device(device_name)
    if requested.type != "cuda":
        return str(requested)

    cuda_index = requested.index if requested.index is not None else 0
    if torch.cuda.is_available() and cuda_index < torch.cuda.device_count():
        return str(requested)

    print(f"CUDA device {device_name!r} is unavailable; using CPU instead.")
    return "cpu"


def build_encoder(args, num_features, norm=None):
    if args.method in ("AllDeepSets", "AllSetformer"):
        if args.LearnMask and norm is None:
            raise ValueError(
                "LearnMask requires a fixed incidence norm and is not supported "
                "for batched hypergraph classification."
            )
        return SetGNN(
            num_features,
            args.embedding_hidden,
            args,
            norm=norm,
        )

    try:
        encoder_cls = ENCODERS[args.method]
    except KeyError as exc:
        raise ValueError(f"Unsupported encoder: {args.method}") from exc

    return encoder_cls(num_features, args.embedding_hidden, args)


def build_model(args, num_features, num_classes, norm=None):
    encoder = build_encoder(args, num_features, norm=norm)

    if args.task_type == "node_cls":
        return NodePredictor(encoder, num_classes, args)
    if args.task_type == "hg_cls":
        return HyperGPredictor(encoder, num_classes, args)
    if args.task_type == "edge_pred":
        aggregator = EDGE_AGGREGATORS[args.aggr_mode](args)
        return EdgePredictor(encoder, aggregator, args)

    raise ValueError(f"Unsupported task type: {args.task_type}")


def prepare_single_hypergraph(args):
    dataset = HyperDataset(args)
    dataset = data_processing(args, dataset)
    return dataset


def prepare_node_classification(args):
    data = prepare_single_hypergraph(args)
    data = algo_preprocessing(data, args)
    masks = data.generate_random_split(
        train_ratio=args.train_prop,
        val_ratio=args.valid_prop,
        seed=args.seed,
    )
    norm = getattr(data, "norm", None)
    model = build_model(args, data.num_features, data.num_classes, norm=norm)
    return model, data, masks


def _load_edge_split(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _edge_split_path(args):
    return osp.join(
        args.edge_save_dir,
        args.edge_split_mode,
        args.dname,
        f"split_{args.seed}.pt",
    )


def load_or_create_edge_split(data, args):
    split_path = _edge_split_path(args)
    data_dict = _load_edge_split(split_path) if osp.exists(split_path) else None

    if data_dict is None or not edge_split_is_current(data_dict):
        if data_dict is not None:
            protocol = data_dict.get("split_protocol", "missing")
            print(
                f"Regenerating stale edge split ({protocol!r} -> "
                f"{EDGE_TRAND_PROTOCOL!r})."
            )
        generate_split_hyperedges(data, args, args.seed)
        data_dict = _load_edge_split(split_path)
        # Split generation uses its own deterministic seed. Restore the run
        # seed so first-time generation and loading an existing cache initialize
        # the model and epoch samplers identically.
        fix_seed(args.seed)

    return data_dict


def _refresh_allset_norm(data, args):
    if args.method not in ("AllDeepSets", "AllSetformer"):
        return

    holder = type("IncidenceData", (), {})()
    holder.edge_index = data.hyperedge_index
    holder = norm_contruction(holder, option=args.normtype)
    data.norm = holder.norm.to(args.device)


def prepare_edge_prediction(args):
    # Self-loop hyperedges are an encoder aid, not positive prediction targets.
    # Build the semantic split before adding them to the observed context.
    data_args = copy(args)
    data_args.add_self_loop = False
    data = prepare_single_hypergraph(data_args)
    data_dict = load_or_create_edge_split(data, args)

    observed_hyperedges = observed_hyperedges_for_embedding(data_dict)
    if not observed_hyperedges:
        raise ValueError("The edge split contains no observed training hyperedges.")

    data.hyperedge_index = build_hyperedge_index_from_hyperedges(
        observed_hyperedges,
        device=args.device,
    )
    data.num_hyperedges = len(observed_hyperedges)
    if args.add_self_loop:
        data.hyperedge_index = add_self_loop_hyperedges(
            data.hyperedge_index,
            data.num_nodes,
        )
        data.num_hyperedges += data.num_nodes
    _refresh_allset_norm(data, args)
    data = algo_preprocessing(data, args)

    batch_loaders = generate_edge_loaders(data_dict, args)
    norm = getattr(data, "norm", None)
    model = build_model(args, data.num_features, data.num_classes, norm=norm)
    return model, data, batch_loaders


def prepare_hypergraph_classification(args):
    dataset = HyperDataset(args).multi_hypergraphs
    train_set, val_set, test_set = generate_split_hypergraphs(
        dataset,
        args.train_prop,
        args.valid_prop,
        args.seed,
    )
    batch_loaders = generate_hg_loaders(train_set, val_set, test_set, args)
    model = build_model(args, dataset.num_features, dataset.num_classes)
    return model, dataset, batch_loaders


def prepare_experiment(args):
    if args.task_type == "node_cls":
        return prepare_node_classification(args)
    if args.task_type == "edge_pred":
        return prepare_edge_prediction(args)
    if args.task_type == "hg_cls":
        return prepare_hypergraph_classification(args)
    raise ValueError(f"Unsupported task type: {args.task_type}")


def run_experiment(args, run_number, total_runs):
    fix_seed(args.seed)

    print(f"\nRun {run_number}/{total_runs} | Seed: {args.seed}")
    model, data, split = prepare_experiment(args)
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable parameters: {num_parameters:,}")

    trainer = Trainer(args)
    model = trainer.training(
        model,
        data,
        args,
        seed_split=split,
        task_type=args.task_type,
    )

    print("Final evaluation")
    evaluator = Evaluator(args)
    return evaluator.evaluate(
        model,
        data,
        seed_split=split,
        task_type=args.task_type,
        verbose=True,
    )


def _mean_std(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    return values.mean(dim=0), values.std(dim=0, unbiased=False)


def summarize_classification_results(results, seeds):
    split_names = ("Train", "Valid", "Test")
    metric_names = {"acc": "Accuracy", "macro_f1": "Macro-F1"}
    summary = {}

    print(f"\n{len(results)}-run summary (mean +/- std) | Seeds: {seeds}")
    for metric, display_name in metric_names.items():
        values = [result[metric] for result in results]
        mean, std = _mean_std(values)
        summary[metric] = {"mean": mean.tolist(), "std": std.tolist()}

        print(display_name)
        for index, split_name in enumerate(split_names):
            print(
                f"  {split_name}: {100 * mean[index]:.2f}% "
                f"+/- {100 * std[index]:.2f}%"
            )

    return summary


def summarize_edge_results(results, seeds):
    selected_metrics = {
        "train": ("roc_train", "ap_train"),
        "val": ("roc_average", "ap_average"),
        "test": ("roc_average", "ap_average"),
    }
    summary = {}

    print(f"\n{len(results)}-run summary (mean +/- std) | Seeds: {seeds}")
    for split, metrics in selected_metrics.items():
        summary[split] = {}
        print(split.capitalize())
        for metric in metrics:
            values = [result[split][metric] for result in results]
            mean, std = _mean_std(values)
            summary[split][metric] = {
                "mean": mean.item(),
                "std": std.item(),
            }
            print(f"  {metric}: {mean.item():.4f} +/- {std.item():.4f}")

    return summary


def main(argv=None):
    args = parameter_parser(argv)
    args.device = resolve_device(args.device)

    print(
        f"Task: {args.task_type} | Dataset: {args.dname} | "
        f"Encoder: {args.method} | Device: {args.device} | Runs: {args.runs}"
    )

    seeds = [args.seed + offset for offset in range(args.runs)]
    results = []
    for run_number, seed in enumerate(seeds, start=1):
        run_args = copy(args)
        run_args.seed = seed
        results.append(run_experiment(run_args, run_number, args.runs))

    if args.runs == 1:
        return results[0]

    if args.task_type in ("node_cls", "hg_cls"):
        summary = summarize_classification_results(results, seeds)
    else:
        summary = summarize_edge_results(results, seeds)

    return {"seeds": seeds, "runs": results, "summary": summary}


if __name__ == "__main__":
    main()
