import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

import test_multitask as base
from dataset.data_base import HyperDataset
from main import build_encoder, load_or_create_edge_split, prepare_single_hypergraph, resolve_device
from MTL.adapter import TaskAdapterBank
from MTL.multitask_agent import MultiTaskExpAgent
from parameter_parser_normal import parameter_parser
from utils.unified_head import UnifiedPredictionHead
from utils.utils import fix_seed, x_svd


EVAL_SPLITS = ("train", "valid", "test")
EDGE_NEGATIVE_TYPES = ("sns", "mns", "cns")
DEFAULT_CONFIG_PATH = "config/multitask_unified.yaml"


def load_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Unified multi-task training")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--node-dataset")
    parser.add_argument("--edge-dataset")
    parser.add_argument("--graph-dataset")
    parser.add_argument("--tasks")
    parser.add_argument("--device")
    parser.add_argument("--encoder")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--feature-dim", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--adapter-hidden", type=int)
    parser.add_argument("--adapter-dropout", type=float)
    parser.add_argument("--adapter-scale", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--node-weight", type=float)
    parser.add_argument("--edge-weight", type=float)
    parser.add_argument("--graph-weight", type=float)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config.setdefault("temperature", 0.5)
    overrides = {name: value for name, value in vars(args).items() if name != "config" and value is not None}
    config.update(overrides)
    return SimpleNamespace(**config)


def selected_tasks(text):
    return [task.strip() for task in text.split(",") if task.strip()]


def project_args(args, task, dataset, seed):
    options = {
        "task": task,
        "dataset": dataset,
        "encoder": args.encoder,
        "device": "cpu",
        "seed": seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight-decay": args.weight_decay,
        "dropout": args.dropout,
        "train-ratio": args.train_ratio,
        "val-ratio": args.val_ratio,
        "hidden-dim": args.hidden_dim,
        "num-layers": args.num_layers,
        "edge-batch-size": args.edge_batch_size,
        "graph-batch-size": args.graph_batch_size,
        "negative-sampling": args.negative_sampling,
        "edge-pooling": args.edge_pooling,
        "pooling": args.graph_pooling,
    }
    argv = [item for name, value in options.items() for item in (f"--{name}", str(value))]
    task_args = parameter_parser(argv + ["--no-self-loops"])
    if args.edge_save_dir:
        task_args.edge_save_dir = args.edge_save_dir
    return task_args


def build_args_by_task(args, seed):
    datasets = {"node_cls": args.node_dataset, "edge_pred": args.edge_dataset, "hg_cls": args.graph_dataset}
    return {task: project_args(args, task, dataset, seed) for task, dataset in datasets.items()}


def load_aligned_data(args):
    task_args = build_args_by_task(args, args.seed)
    node_data = prepare_single_hypergraph(task_args["node_cls"])
    edge_data = node_data if args.node_dataset == args.edge_dataset else prepare_single_hypergraph(task_args["edge_pred"])
    graph_data = HyperDataset(task_args["hg_cls"]).multi_hypergraphs

    feature_dim = args.feature_dim or max(node_data.num_features, edge_data.num_features, graph_data.num_features)
    node_data.x = x_svd(node_data.x, feature_dim).float()
    node_data.num_features = feature_dim
    if edge_data is not node_data:
        edge_data.x = x_svd(edge_data.x, feature_dim).float()
        edge_data.num_features = feature_dim
    return node_data, edge_data, graph_data, feature_dim


def build_loaders(args, tasks, seed, node_data, edge_data, graph_data, feature_dim, edge_args):
    node_loaders = base.build_node_loaders(node_data, args, seed)
    edge_dict = load_or_create_edge_split(edge_data, edge_args)
    edge_state = base.build_edge_state(edge_data, edge_dict, args)
    graph_loaders = base.build_graph_loaders(graph_data, feature_dim, args, seed)

    loaders = {"node_cls": node_loaders["train"], "edge_pred": edge_state["train"], "hg_cls": graph_loaders["train"]}
    train_loaders = {task: loaders[task] for task in tasks}
    return node_loaders, edge_state, graph_loaders, train_loaders


def build_model(encoder, node_data, graph_data, task_args, tasks, args):
    label_spaces = {}
    if "node_cls" in tasks:
        label_spaces["node_cls"] = node_data.num_classes
    if "edge_pred" in tasks:
        label_spaces["edge_pred"] = 2
    if "hg_cls" in tasks:
        label_spaces["hg_cls"] = graph_data.num_classes

    embedding_dim = task_args["node_cls"].embedding_hidden
    adapter_bank = TaskAdapterBank(embedding_dim, tasks, args.adapter_hidden,
                                   args.adapter_dropout, args.adapter_scale)

    return UnifiedPredictionHead(encoder, label_spaces, embedding_dim,
                                 edge_pooling=task_args["edge_pred"].aggr_mode,
                                 graph_pooling=task_args["hg_cls"].pooling,
                                 temperature=args.temperature,
                                 adapter_bank=adapter_bank)


def build_agent(args, device):
    agent_args = SimpleNamespace(device=device, lr=args.lr, wd=args.weight_decay,
                                 epochs=args.epochs, display_step=args.eval_every)
    weights = {"node_cls": args.node_weight, "edge_pred": args.edge_weight, "hg_cls": args.graph_weight}
    return MultiTaskExpAgent(agent_args, weights)


@torch.no_grad()
def evaluate_classification(model, loader, device):
    model.eval()
    predictions, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        predictions.append(model(batch).argmax(dim=-1).cpu())
        labels.append(batch.y.view(-1).cpu())

    y_pred = torch.cat(predictions).numpy()
    y_true = torch.cat(labels).numpy()
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_f1": f1_score(y_true, y_pred, average="macro")}


@torch.no_grad()
def predict_edges(model, loader, device):
    model.eval()
    scores = []
    for batch in loader:
        batch = batch.to(device)
        scores.append(torch.softmax(model(batch), dim=-1)[:, 1].cpu())
    return torch.cat(scores).numpy()


def evaluate_edge_split(model, edge_state, split, batch_size, device):
    adapter = edge_state["adapter"]
    data_dict = edge_state["data_dict"]
    positives = base.split_positive_hyperedges(data_dict, split)
    positive_scores = predict_edges(model, base.EdgeEvalTaskLoader(adapter, positives, 1, batch_size), device)
    negative_scores = {
        name: predict_edges(model, base.EdgeEvalTaskLoader(adapter, data_dict[f"{split}_{name}"], 0, batch_size), device)
        for name in EDGE_NEGATIVE_TYPES
    }

    metrics = {}
    for name, scores in negative_scores.items():
        roc, ap, accuracy = base.binary_metrics(positive_scores, scores)
        metrics.update({f"roc_{name}": roc, f"ap_{name}": ap, f"accuracy_{name}": accuracy})

    mixed_size = len(positive_scores) // len(EDGE_NEGATIVE_TYPES)
    mixed_scores = np.concatenate([negative_scores[name][:mixed_size] for name in EDGE_NEGATIVE_TYPES])
    roc, ap, accuracy = base.binary_metrics(positive_scores, mixed_scores)
    metrics.update({"roc_mixed": roc, "ap_mixed": ap, "accuracy_mixed": accuracy})

    groups = (*EDGE_NEGATIVE_TYPES, "mixed")
    for metric in ("roc", "ap", "accuracy"):
        metrics[f"{metric}_average"] = np.mean([metrics[f"{metric}_{name}"] for name in groups])
    return metrics


def evaluate_run(model, tasks, node_loaders, edge_state, graph_loaders, args, device):
    results = {}
    if "node_cls" in tasks:
        results["node_cls"] = {split: evaluate_classification(model, node_loaders[split], device) for split in EVAL_SPLITS}
    if "edge_pred" in tasks:
        results["edge_pred"] = {split: evaluate_edge_split(model, edge_state, split, args.edge_batch_size, device) for split in EVAL_SPLITS}
    if "hg_cls" in tasks:
        results["hg_cls"] = {split: evaluate_classification(model, graph_loaders[split], device) for split in EVAL_SPLITS}
    return results


def print_results(results):
    for task, splits in results.items():
        for split, metrics in splits.items():
            if task == "edge_pred":
                print(f"{task} {split:>5} | Accuracy {metrics['accuracy_average']:.4f} | AUROC {metrics['roc_average']:.4f} | AP {metrics['ap_average']:.4f}")
            else:
                print(f"{task} {split:>5} | Accuracy {metrics['accuracy']:.4f} | Macro-F1 {metrics['macro_f1']:.4f}")


def print_summary(run_results, tasks):
    print("\nRuns summary (mean +/- std)")
    metrics = {
        "node_cls": ("accuracy", "macro_f1"),
        "edge_pred": ("roc_average", "ap_average"),
        "hg_cls": ("accuracy", "macro_f1"),
    }
    for task in tasks:
        name1, name2 = metrics[task]
        values1 = np.array([result[task]["test"][name1] for result in run_results])
        values2 = np.array([result[task]["test"][name2] for result in run_results])
        print(f"{task} | {name1}: {values1.mean():.4f} +/- {values1.std():.4f} | {name2}: {values2.mean():.4f} +/- {values2.std():.4f}")


def run_once(args, tasks, seed, device, node_data, edge_data, graph_data, feature_dim):
    fix_seed(seed)
    task_args = build_args_by_task(args, seed)
    node_loaders, edge_state, graph_loaders, train_loaders = build_loaders(
        args, tasks, seed, node_data, edge_data, graph_data, feature_dim, task_args["edge_pred"]
    )

    encoder = build_encoder(task_args["node_cls"], feature_dim + 1)
    model = build_model(encoder, node_data, graph_data, task_args, tasks, args)
    agent = build_agent(args, device)

    print(f"\nRun seed {seed} | " + " | ".join(f"{task}: {len(train_loaders[task])} batches" for task in tasks))
    model = agent.training(model, train_loaders)
    results = evaluate_run(model, tasks, node_loaders, edge_state, graph_loaders, args, device)
    print_results(results)
    return results


def main():
    args = parse_args()
    tasks = selected_tasks(args.tasks)
    device = resolve_device(args.device)

    print(f"Unified multi-task | tasks={tasks} | node={args.node_dataset} | edge={args.edge_dataset} | graph={args.graph_dataset} | device={device} | runs={args.runs}")
    node_data, edge_data, graph_data, feature_dim = load_aligned_data(args)
    print(f"Feature dimension: {feature_dim} + 1 query marker = {feature_dim + 1}")

    run_results = []
    for run in range(args.runs):
        run_results.append(run_once(args, tasks, args.seed + run, device, node_data, edge_data, graph_data, feature_dim))
    print_summary(run_results, tasks)


if __name__ == "__main__":
    main()
