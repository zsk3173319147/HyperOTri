import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch_geometric.loader import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.data_base import HyperDataset
from dataset.edge_loaders import (
    build_hyperedge_index_from_hyperedges,
    observed_hyperedges_for_embedding,
    split_positive_hyperedges,
)
from dataset.hg_loaders import generate_split_hypergraphs
from main import (
    EDGE_AGGREGATORS,
    build_encoder,
    load_or_create_edge_split,
    prepare_single_hypergraph,
    resolve_device,
)

from MTL.subgraph_builder import PropagationSubgraphBuilder
from MTL.task_agent import EdgePredTaskAdapter, GraphClsTaskAdapter, NodeClsTaskAdapter
from parameter_parser_normal import METHODS, MULTI_DATASETS, SINGLE_DATASETS, parameter_parser
from utils.task_heads import EdgePredictor, HyperGPredictor, NodePredictor
from utils.utils import fix_seed, x_svd


TASK_NAMES = ("node_cls", "edge_pred", "hg_cls")
EDGE_NEGATIVE_TYPES = ("sns", "mns", "cns")
EVAL_SPLITS = ("train", "valid", "test")


def parse_args():
    parser = argparse.ArgumentParser(description="使用完整数据训练并评估共享 HGNN 多任务模型。",formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    data = parser.add_argument_group("数据与任务")
    data.add_argument("--node-dataset", choices=SINGLE_DATASETS, default="cora")
    data.add_argument("--edge-dataset", choices=SINGLE_DATASETS, default="citeseer")
    data.add_argument("--graph-dataset", choices=MULTI_DATASETS, default="RHG_3")
    data.add_argument("--tasks", default="node_cls,edge_pred,hg_cls")
    data.add_argument("--feature-dim",type=int,default=0,help="统一特征维度；0 表示取三个数据集原始维度的最大值并补零。",)
    data.add_argument("--train-ratio", type=float, default=0.6)
    data.add_argument("--val-ratio", type=float, default=0.2)
    data.add_argument("--edge-save-dir", default=None)

    training = parser.add_argument_group("训练")
    training.add_argument("--device", default="cuda:0")
    training.add_argument("--epochs", type=int, default=100)
    training.add_argument("--lr", type=float, default=1e-3)
    training.add_argument("--weight-decay", type=float, default=0.0)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--runs", type=int, default=1)
    training.add_argument("--eval-every", type=int, default=20)

    model = parser.add_argument_group("Encoder 与预测头")
    model.add_argument("--encoder", choices=METHODS, default="HGNN")
    model.add_argument("--hidden-dim", type=int, default=128)
    model.add_argument("--num-layers", type=int, default=2)
    model.add_argument("--dropout", type=float, default=0.2)
    model.add_argument("--head-hidden", type=int, default=128)
    model.add_argument("--head-layers", type=int, default=1)
    model.add_argument("--head-dropout", type=float, default=0.2)
    model.add_argument("--head-normalization",choices=("bn", "ln", "None"),default="ln",)

    batch = parser.add_argument_group("任务 batch")
    batch.add_argument("--node-batch-size", type=int, default=128)
    batch.add_argument("--edge-batch-size", type=int, default=128)
    batch.add_argument("--graph-batch-size", type=int, default=32)
    batch.add_argument("--steps-per-epoch", type=int, default=0)

    multitask = parser.add_argument_group("多任务权重")
    multitask.add_argument("--node-weight", type=float, default=1.0)
    multitask.add_argument("--edge-weight", type=float, default=1.0)
    multitask.add_argument("--graph-weight", type=float, default=1.0)

    subgraph = parser.add_argument_group("子超图与 readout")
    subgraph.add_argument("--context-hops", type=int, default=2)
    subgraph.add_argument("--max-hyperedges", type=int, default=16)
    subgraph.add_argument("--negative-sampling",choices=("sns", "mns", "cns", "mixed"),default="mixed",)
    subgraph.add_argument("--edge-pooling",choices=("mean", "max", "maxmin"),default="maxmin",)
    subgraph.add_argument("--graph-pooling",choices=("mean", "max"),default="max",)
    return parser.parse_args()


def selected_tasks(text):
    return [task.strip() for task in text.split(",") if task.strip()]


def project_args(cli_args, task_type, dataset, seed):
    options = {
        "task": task_type,
        "dataset": dataset,
        "encoder": cli_args.encoder,
        "device": "cpu",
        "seed": seed,
        "epochs": cli_args.epochs,
        "lr": cli_args.lr,
        "weight-decay": cli_args.weight_decay,
        "dropout": cli_args.dropout,
        "train-ratio": cli_args.train_ratio,
        "val-ratio": cli_args.val_ratio,
        "hidden-dim": cli_args.hidden_dim,
        "num-layers": cli_args.num_layers,
        "head-hidden": cli_args.head_hidden,
        "head-layers": cli_args.head_layers,
        "head-dropout": cli_args.head_dropout,
        "head-normalization": cli_args.head_normalization,
        "edge-batch-size": cli_args.edge_batch_size,
        "graph-batch-size": cli_args.graph_batch_size,
        "negative-sampling": cli_args.negative_sampling,
        "edge-pooling": cli_args.edge_pooling,
        "pooling": cli_args.graph_pooling,
    }
    argv = [item for key, value in options.items() for item in (f"--{key}", str(value))]
    args = parameter_parser(argv + ["--no-self-loops"])
    if cli_args.edge_save_dir:
        args.edge_save_dir = cli_args.edge_save_dir
    return args


class NodeTaskLoader:
    def __init__(self, adapter, node_ids, batch_size, shuffle):
        self.adapter = adapter
        self.node_ids = list(node_ids)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(len(self.node_ids) / self.batch_size)

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(len(self.node_ids)).tolist()
            node_ids = [self.node_ids[index] for index in order]
        else:
            node_ids = self.node_ids

        for start in range(0, len(node_ids), self.batch_size):
            yield self.adapter.build_batch(node_ids[start : start + self.batch_size])


class EdgeTrainTaskLoader:
    """每轮使用全部正样本，并从负样本池随机取等量负样本。"""

    def __init__(self, adapter, positives, negative_pool, batch_size):
        self.adapter = adapter
        self.positives = list(positives)
        self.negative_pool = list(negative_pool)
        self.batch_size = batch_size
        self.positive_batch_size = max(1, batch_size // 2)

    def __len__(self):
        return math.ceil(len(self.positives) / self.positive_batch_size)

    def __iter__(self):
        positive_order = torch.randperm(len(self.positives)).tolist()
        positives = [self.positives[index] for index in positive_order]

        if len(self.negative_pool) >= len(positives):
            negative_ids = torch.randperm(len(self.negative_pool))[: len(positives)]
        else:
            negative_ids = torch.randint(
                len(self.negative_pool),
                (len(positives),),
            )
        negatives = [self.negative_pool[index] for index in negative_ids.tolist()]

        for start in range(0, len(positives), self.positive_batch_size):
            positive_batch = positives[start : start + self.positive_batch_size]
            negative_batch = negatives[start : start + len(positive_batch)]
            candidates = positive_batch + negative_batch
            labels = [1.0] * len(positive_batch) + [0.0] * len(negative_batch)
            yield self.adapter.build_batch(candidates, labels)


class EdgeEvalTaskLoader:
    def __init__(self, adapter, candidates, label, batch_size):
        self.adapter = adapter
        self.candidates = list(candidates)
        self.label = float(label)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.candidates) / self.batch_size)

    def __iter__(self):
        for start in range(0, len(self.candidates), self.batch_size):
            candidates = self.candidates[start : start + self.batch_size]
            labels = [self.label] * len(candidates)
            yield self.adapter.build_batch(candidates, labels)


class GraphTaskLoader:
    def __init__(self, graphs, feature_dim, batch_size, shuffle):
        self.loader = DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)
        self.feature_dim = feature_dim
        self.adapter = GraphClsTaskAdapter()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for graph_batch in self.loader:
            graph_batch.x = x_svd(graph_batch.x, self.feature_dim).float()
            yield self.adapter.build_batch(graph_batch)


def mask_indices(masks):
    return {
        split: mask.nonzero(as_tuple=False).view(-1).tolist()
        for split, mask in masks.items()
    }


def build_node_loaders(data, cli_args, seed):
    masks = data.generate_random_split(
        train_ratio=cli_args.train_ratio,
        val_ratio=cli_args.val_ratio,
        seed=seed,
    )
    indices = mask_indices(masks)
    builder = PropagationSubgraphBuilder(
        data.x,
        data.hyperedge_index,
        context_hops=cli_args.context_hops,
        max_hyperedges=cli_args.max_hyperedges,
    )
    adapter = NodeClsTaskAdapter(builder, data.y)
    return {
        split: NodeTaskLoader(
            adapter,
            node_ids,
            cli_args.node_batch_size,
            shuffle=split == "train",
        )
        for split, node_ids in indices.items()
    }


def mixed_negative_hyperedges(data_dict):
    negatives = []
    seen = set()
    for name in ("train_sns", "train_mns", "train_cns"):
        for hyperedge in data_dict[name]:
            key = frozenset(hyperedge)
            if key not in seen:
                negatives.append(hyperedge)
                seen.add(key)
    return negatives


def build_edge_state(data, data_dict, cli_args):
    context_index = build_hyperedge_index_from_hyperedges(
        observed_hyperedges_for_embedding(data_dict)
    )
    builder = PropagationSubgraphBuilder(
        data.x,
        context_index,
        context_hops=cli_args.context_hops,
        max_hyperedges=cli_args.max_hyperedges,
    )
    adapter = EdgePredTaskAdapter(builder)
    positives = split_positive_hyperedges(data_dict, "train")
    if cli_args.negative_sampling == "mixed":
        negative_pool = mixed_negative_hyperedges(data_dict)
    else:
        negative_pool = data_dict[f"train_{cli_args.negative_sampling}"]

    return {
        "adapter": adapter,
        "train": EdgeTrainTaskLoader(
            adapter,
            positives,
            negative_pool,
            cli_args.edge_batch_size,
        ),
        "data_dict": data_dict,
    }


def build_graph_loaders(dataset, feature_dim, cli_args, seed):
    train_set, valid_set, test_set = generate_split_hypergraphs(
        dataset,
        cli_args.train_ratio,
        cli_args.val_ratio,
        seed,
    )
    return {
        "train": GraphTaskLoader(
            train_set,
            feature_dim,
            cli_args.graph_batch_size,
            shuffle=True,
        ),
        "valid": GraphTaskLoader(
            valid_set,
            feature_dim,
            cli_args.graph_batch_size,
            shuffle=False,
        ),
        "test": GraphTaskLoader(
            test_set,
            feature_dim,
            cli_args.graph_batch_size,
            shuffle=False,
        ),
    }


@torch.no_grad()
def evaluate_classification(agent, task_models, loader, device):
    for model in task_models.values():
        model.eval()
    predictions = []
    labels = []
    for batch in loader:
        batch = batch.to(device)
        logits = agent.forward_task(task_models, batch)
        predictions.append(logits.argmax(dim=-1).cpu())
        labels.append(batch.y.view(-1).cpu())

    y_pred = torch.cat(predictions).numpy()
    y_true = torch.cat(labels).numpy()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


@torch.no_grad()
def predict_edge(agent, task_models, loader, device):
    for model in task_models.values():
        model.eval()
    scores = []
    for batch in loader:
        batch = batch.to(device)
        logits = agent.forward_task(task_models, batch)
        scores.append(torch.sigmoid(logits).cpu())
    return torch.cat(scores).numpy()


def binary_metrics(positive_scores, negative_scores):
    labels = np.concatenate(
        [np.ones(len(positive_scores)), np.zeros(len(negative_scores))]
    )
    scores = np.concatenate([positive_scores, negative_scores])
    predictions = (scores >= 0.5).astype(int)
    return (
        roc_auc_score(labels, scores),
        average_precision_score(labels, scores),
        accuracy_score(labels, predictions),
    )


def predict_edge_candidates(agent, task_models, adapter, candidates, label, batch_size, device):
    loader = EdgeEvalTaskLoader(adapter, candidates, label, batch_size)
    return predict_edge(agent, task_models, loader, device)


def add_binary_metrics(metrics, name, positive_scores, negative_scores):
    roc, ap, accuracy = binary_metrics(positive_scores, negative_scores)
    metrics[f"roc_{name}"] = roc
    metrics[f"ap_{name}"] = ap
    metrics[f"accuracy_{name}"] = accuracy


def evaluate_edge_split(agent, task_models, edge_state, split, batch_size, device):
    adapter = edge_state["adapter"]
    data_dict = edge_state["data_dict"]
    positive_scores = predict_edge_candidates(agent, task_models, adapter, split_positive_hyperedges(data_dict, split), 1, batch_size, device)

    negative_scores = {}
    for method in EDGE_NEGATIVE_TYPES:
        negative_scores[method] = predict_edge_candidates(agent, task_models, adapter, data_dict[f"{split}_{method}"], 0, batch_size, device)

    # 分别看三种负采样，再用 mixed 汇总，和单任务 edge_pred 的评估口径保持一致。
    metrics = {}
    for method in EDGE_NEGATIVE_TYPES:
        add_binary_metrics(metrics, method, positive_scores, negative_scores[method])

    mixed_size = len(positive_scores) // 3
    mixed_scores = np.concatenate(
        [negative_scores[method][:mixed_size] for method in EDGE_NEGATIVE_TYPES]
    )
    add_binary_metrics(metrics, "mixed", positive_scores, mixed_scores)

    score_groups = (*EDGE_NEGATIVE_TYPES, "mixed")
    for metric_name in ("roc", "ap", "accuracy"):
        values = [metrics[f"{metric_name}_{name}"] for name in score_groups]
        metrics[f"{metric_name}_average"] = np.mean(values)

    return metrics


def build_task_models(encoder, node_data, graph_data, args_by_task, tasks):
    models = {
        "node_cls": NodePredictor(
            encoder,
            node_data.num_classes,
            args_by_task["node_cls"],
        ),
        "edge_pred": EdgePredictor(
            encoder,
            EDGE_AGGREGATORS[args_by_task["edge_pred"].aggr_mode](
                args_by_task["edge_pred"]
            ),
            args_by_task["edge_pred"],
        ),
        "hg_cls": HyperGPredictor(
            encoder,
            graph_data.num_classes,
            args_by_task["hg_cls"],
        ),
    }
    return {task: models[task] for task in tasks}


def print_run_results(results):
    if "node_cls" in results:
        for split, metrics in results["node_cls"].items():
            print(
                f"node_cls {split:>5} | Accuracy {metrics['accuracy']:.4f} | "
                f"Macro-F1 {metrics['macro_f1']:.4f}"
            )
    if "edge_pred" in results:
        for split, metrics in results["edge_pred"].items():
            print(
                f"edge_pred {split:>5} | Accuracy {metrics['accuracy_average']:.4f} | "
                f"AUROC {metrics['roc_average']:.4f} | "
                f"AP {metrics['ap_average']:.4f}"
            )
    if "hg_cls" in results:
        for split, metrics in results["hg_cls"].items():
            print(
                f"hg_cls {split:>5} | Accuracy {metrics['accuracy']:.4f} | "
                f"Macro-F1 {metrics['macro_f1']:.4f}"
            )


def load_aligned_data(cli_args):
    load_args = build_args_by_task(cli_args, cli_args.seed)
    node_data = prepare_single_hypergraph(load_args["node_cls"])
    if cli_args.node_dataset == cli_args.edge_dataset:
        edge_data = node_data
    else:
        edge_data = prepare_single_hypergraph(load_args["edge_pred"])
    graph_data = HyperDataset(load_args["hg_cls"]).multi_hypergraphs
    if cli_args.feature_dim:
        feature_dim = cli_args.feature_dim
    else:
        feature_dim = max(
            node_data.num_features,
            edge_data.num_features,
            graph_data.num_features,
        )
    node_data.x = x_svd(node_data.x, feature_dim).float()
    node_data.num_features = feature_dim
    if edge_data is not node_data:
        edge_data.x = x_svd(edge_data.x, feature_dim).float()
        edge_data.num_features = feature_dim
    return node_data, edge_data, graph_data, feature_dim


def build_args_by_task(cli_args, seed):
    datasets = {
        "node_cls": cli_args.node_dataset,
        "edge_pred": cli_args.edge_dataset,
        "hg_cls": cli_args.graph_dataset,
    }
    return {task: project_args(cli_args, task, dataset, seed) for task, dataset in datasets.items()}


def build_loaders(cli_args, tasks, seed, node_data, edge_data, graph_data, feature_dim, edge_args):
    node_loaders = build_node_loaders(node_data, cli_args, seed)
    edge_dict = load_or_create_edge_split(edge_data, edge_args)
    edge_state = build_edge_state(edge_data, edge_dict, cli_args)
    graph_loaders = build_graph_loaders(graph_data, feature_dim, cli_args, seed)

    all_train_loaders = {
        "node_cls": node_loaders["train"],
        "edge_pred": edge_state["train"],
        "hg_cls": graph_loaders["train"],
    }
    train_loaders = {task: all_train_loaders[task] for task in tasks}
    return node_loaders, edge_state, graph_loaders, train_loaders


def build_agent(cli_args, device):
    agent_args = SimpleNamespace(
        device=device,
        lr=cli_args.lr,
        wd=cli_args.weight_decay,
        epochs=cli_args.epochs,
        display_step=cli_args.eval_every,
        steps_per_epoch=cli_args.steps_per_epoch,
    )
    task_weights = {
        "node_cls": cli_args.node_weight,
        "edge_pred": cli_args.edge_weight,
        "hg_cls": cli_args.graph_weight,
    }
    return NativeMultiTaskExpAgent(agent_args, task_weights=task_weights)


def evaluate_run(agent, task_models, tasks, node_loaders, edge_state, graph_loaders, cli_args, device):
    results = {}
    if "node_cls" in tasks:
        results["node_cls"] = {split: evaluate_classification(agent, task_models, node_loaders[split], device) for split in EVAL_SPLITS}
    if "edge_pred" in tasks:
        results["edge_pred"] = {split: evaluate_edge_split(agent, task_models, edge_state, split, cli_args.edge_batch_size, device) for split in EVAL_SPLITS}
    if "hg_cls" in tasks:
        results["hg_cls"] = {split: evaluate_classification(agent, task_models, graph_loaders[split], device) for split in EVAL_SPLITS}
    return results


def print_run_start(seed, feature_dim, tasks, train_loaders):
    print(f"\nRun seed {seed} | feature_dim={feature_dim}")
    print("Train batches | " + " | ".join(f"{task}: {len(train_loaders[task])}" for task in tasks))


def print_experiment_header(cli_args, tasks, device):
    print(f"Multi-task training | Tasks: {tasks} | node={cli_args.node_dataset} | edge={cli_args.edge_dataset} | graph={cli_args.graph_dataset} | {cli_args.encoder} | Device: {device} | Runs: {cli_args.runs}")


def run_once(cli_args, tasks, seed, device, node_data, edge_data, graph_data, feature_dim):
    fix_seed(seed)

    args_by_task = build_args_by_task(cli_args, seed)
    node_loaders, edge_state, graph_loaders, train_loaders = build_loaders(cli_args, tasks, seed, node_data, edge_data, graph_data, feature_dim, args_by_task["edge_pred"])
    encoder = build_encoder(args_by_task["node_cls"], feature_dim + 1)
    task_models = build_task_models(encoder, node_data, graph_data, args_by_task, tasks)
    agent = build_agent(cli_args, device)

    print_run_start(seed, feature_dim, tasks, train_loaders)
    task_models = agent.training(task_models, train_loaders)

    results = evaluate_run(agent, task_models, tasks, node_loaders, edge_state, graph_loaders, cli_args, device)
    print_run_results(results)
    return results


def print_summary(run_results, tasks):
    print("\nRuns summary (mean +/- std)")
    if "node_cls" in tasks:
        accuracy = np.asarray(
            [result["node_cls"]["test"]["accuracy"] for result in run_results]
        )
        macro_f1 = np.asarray(
            [result["node_cls"]["test"]["macro_f1"] for result in run_results]
        )
        print(
            f"node_cls Test Accuracy: {accuracy.mean():.4f} +/- {accuracy.std():.4f} | "
            f"Macro-F1: {macro_f1.mean():.4f} +/- {macro_f1.std():.4f}"
        )
    if "edge_pred" in tasks:
        auroc = np.asarray(
            [result["edge_pred"]["test"]["roc_average"] for result in run_results]
        )
        ap = np.asarray(
            [result["edge_pred"]["test"]["ap_average"] for result in run_results]
        )
        print(
            f"edge_pred Test AUROC: {auroc.mean():.4f} +/- {auroc.std():.4f} | "
            f"AP: {ap.mean():.4f} +/- {ap.std():.4f}"
        )
    if "hg_cls" in tasks:
        accuracy = np.asarray(
            [result["hg_cls"]["test"]["accuracy"] for result in run_results]
        )
        macro_f1 = np.asarray(
            [result["hg_cls"]["test"]["macro_f1"] for result in run_results]
        )
        print(
            f"hg_cls Test Accuracy: {accuracy.mean():.4f} +/- {accuracy.std():.4f} | "
            f"Macro-F1: {macro_f1.mean():.4f} +/- {macro_f1.std():.4f}"
        )


def main():
    cli_args = parse_args()
    tasks = selected_tasks(cli_args.tasks)
    device = resolve_device(cli_args.device)

    print_experiment_header(cli_args, tasks, device)
    node_data, edge_data, graph_data, feature_dim = load_aligned_data(cli_args)
    print(f"Aligned feature dimension: {feature_dim}")

    run_results = []
    for run in range(cli_args.runs):
        result = run_once(cli_args, tasks, cli_args.seed + run, device, node_data, edge_data, graph_data, feature_dim)
        run_results.append(result)

    print_summary(run_results, tasks)


if __name__ == "__main__":
    main()
