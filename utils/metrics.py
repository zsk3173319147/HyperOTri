from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from HNN.preprocessing import algo_preprocessing
from utils.utils import add_self_loop_hyperedges, relabel_hyperedge_index


def eval_acc(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1).detach().cpu().numpy()
    is_labeled = y_true == y_true
    return float(np.mean(y_true[is_labeled] == y_pred[is_labeled]))


@torch.no_grad()
def evaluate(model, data, split_idx, result=None):
    if result is None:
        model.eval()
        result = F.log_softmax(model(data), dim=1)

    train_acc = eval_acc(data.y[split_idx["train"]], result[split_idx["train"]])
    valid_acc = eval_acc(data.y[split_idx["valid"]], result[split_idx["valid"]])
    test_acc = eval_acc(data.y[split_idx["test"]], result[split_idx["test"]])
    return train_acc, valid_acc, test_acc


def evaluate_hypegraph(model, batch_loaders, args):
    metrics = defaultdict(list)
    for _, loader in batch_loaders.items():
        acc, macro_f1 = test_hypegraph_loader(model, loader, args)
        metrics["acc"].append(acc)
        metrics["macro_f1"].append(macro_f1)
    return metrics


def test_hypegraph_loader(model, data_loader, args):
    model.eval()
    predictions, labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            batch.hyperedge_index, _ = relabel_hyperedge_index(batch.hyperedge_index)

            if args.method == "HyperGCN":
                encoder = getattr(model, "encoder", model)
                if hasattr(encoder, "structure"):
                    encoder.structure = None
                batch.hyperedge_index = add_self_loop_hyperedges(batch.hyperedge_index, batch.num_nodes)

            batch = algo_preprocessing(batch, args)
            if args.method in ["AllSetformer", "AllDeepSets"]:
                batch.norm = torch.ones(batch.hyperedge_index.size(1), dtype=batch.x.dtype, device=batch.hyperedge_index.device)

            batch = batch.to(args.device)
            predictions.append(model(batch).argmax(dim=1).cpu())
            labels.append(batch.y.cpu())

    predictions = torch.cat(predictions).numpy()
    labels = torch.cat(labels).numpy()
    return accuracy_score(labels, predictions), f1_score(labels, predictions, average="macro")


def hg_evaluation_printer(result):
    print(f'Train Acc: {100 * result["acc"][0]:.2f}%, Valid Acc: {100 * result["acc"][1]:.2f}%, Test  Acc: {100 * result["acc"][2]:.2f}%')
    print(f'Train Macro-F1: {100 * result["macro_f1"][0]:.2f}%, Valid Macro-F1: {100 * result["macro_f1"][1]:.2f}%, Test  Macro-F1: {100 * result["macro_f1"][2]:.2f}%')


def evaluate_edge(model, data, batch_loaders):
    train_metrics = eval_edge_train(model, data, batch_loaders)
    val_metrics = eval_edge_val_test(model, data, batch_loaders, mode="val")
    test_metrics = eval_edge_val_test(model, data, batch_loaders, mode="test")
    return train_metrics, val_metrics, test_metrics


def edge_evaluation_printer(train_metrics, val_metrics, test_metrics):
    print("---------Train set-----------")
    print(f"|AUROC| {train_metrics['roc_train']:.4f}")
    print(f"|AP| {train_metrics['ap_train']:.4f}")
    print("---------Valid set-----------")
    print(f"|AUROC| SNS: {val_metrics['roc_sns']:.4f} | MNS: {val_metrics['roc_mns']:.4f} | CNS: {val_metrics['roc_cns']:.4f} | MIX: {val_metrics['roc_mixed']:.4f} | AVG: {val_metrics['roc_average']:.4f}")
    print(f"|AP| SNS: {val_metrics['ap_sns']:.4f} | MNS: {val_metrics['ap_mns']:.4f} | CNS: {val_metrics['ap_cns']:.4f} | MIX: {val_metrics['ap_mixed']:.4f} | AVG: {val_metrics['ap_average']:.4f}")
    print("---------Test set-----------")
    print(f"|AUROC| SNS: {test_metrics['roc_sns']:.4f} | MNS: {test_metrics['roc_mns']:.4f} | CNS: {test_metrics['roc_cns']:.4f} | MIX: {test_metrics['roc_mixed']:.4f} | AVG: {test_metrics['roc_average']:.4f}")
    print(f"|AP| SNS: {test_metrics['ap_sns']:.4f} | MNS: {test_metrics['ap_mns']:.4f} | CNS: {test_metrics['ap_cns']:.4f} | MIX: {test_metrics['ap_mixed']:.4f} | AVG: {test_metrics['ap_average']:.4f}")


def aggr_metrics(metrics_dict, result):
    for split, metrics in result.items():
        for name, value in metrics.items():
            metrics_dict[split][name].append(value)
    return metrics_dict


def avg_result_printer_edge(metrics_dict):
    train_mean = [np.mean(values) for values in metrics_dict["train"].values()]
    train_std = [np.std(values) for values in metrics_dict["train"].values()]
    val_mean = [np.mean(values) for values in metrics_dict["val"].values()]
    val_std = [np.std(values) for values in metrics_dict["val"].values()]
    test_mean = [np.mean(values) for values in metrics_dict["test"].values()]
    test_std = [np.std(values) for values in metrics_dict["test"].values()]

    print("---------Train set-----------")
    print(f"|AUROC| {train_mean[0]:.4f}+-{train_std[0]:.4f}")
    print(f"|AP| {train_mean[1]:.4f}+-{train_std[1]:.4f}")
    print("---------Valid set-----------")
    print(f"|AUROC| SNS: {val_mean[0]:.4f}+-{val_std[0]:.4f} | MNS: {val_mean[2]:.4f}+-{val_std[2]:.4f} | CNS: {val_mean[4]:.4f}+-{val_std[4]:.4f} | MIX: {val_mean[6]:.4f}+-{val_std[6]:.4f} | AVG: {val_mean[8]:.4f}+-{val_std[8]:.4f}")
    print(f"|AP| SNS: {val_mean[1]:.4f}+-{val_std[1]:.4f} | MNS: {val_mean[3]:.4f}+-{val_std[3]:.4f} | CNS: {val_mean[5]:.4f}+-{val_std[5]:.4f} | MIX: {val_mean[7]:.4f}+-{val_std[7]:.4f} | AVG: {val_mean[9]:.4f}+-{val_std[9]:.4f}")
    print("---------Test set-----------")
    print(f"|AUROC| SNS: {test_mean[0]:.4f}+-{test_std[0]:.4f} | MNS: {test_mean[2]:.4f}+-{test_std[2]:.4f} | CNS: {test_mean[4]:.4f}+-{test_std[4]:.4f} | MIX: {test_mean[6]:.4f}+-{test_std[6]:.4f} | AVG: {test_mean[8]:.4f}+-{test_std[8]:.4f}")
    print(f"|AP| SNS: {test_mean[1]:.4f}+-{test_std[1]:.4f} | MNS: {test_mean[3]:.4f}+-{test_std[3]:.4f} | CNS: {test_mean[5]:.4f}+-{test_std[5]:.4f} | MIX: {test_mean[7]:.4f}+-{test_std[7]:.4f} | AVG: {test_mean[9]:.4f}+-{test_std[9]:.4f}")


def evaluate_edge_loader(model, data, dataloader):
    model.eval()
    predictions, labels = [], []
    train_mode = getattr(dataloader, "test_generator", None)

    if hasattr(dataloader, "eval"):
        dataloader.eval()
    iter(dataloader)

    with torch.no_grad():
        node_embeddings, _ = model.encoding(data)
        while True:
            hyperedges, batch_labels, is_last = dataloader.next()
            predictions += model.aggregate(node_embeddings, hyperedges, mode="Eval")
            labels.append(batch_labels.detach())
            if is_last:
                break

    if train_mode is False and hasattr(dataloader, "train"):
        dataloader.train()
    if hasattr(dataloader, "_cursor"):
        dataloader._cursor = 0

    predictions = torch.sigmoid(torch.stack(predictions).reshape(-1)).tolist()
    labels = torch.cat(labels).tolist()
    return predictions, labels


def eval_edge_train(model, data, batch_loaders):
    pred_pos, label_pos = evaluate_edge_loader(model, data, batch_loaders["train_pos"])
    pred_neg, label_neg = evaluate_edge_loader(model, data, batch_loaders["train_neg"])
    labels = np.array(label_pos + label_neg)
    predictions = np.array(pred_pos + pred_neg)
    return {"roc_train": roc_auc_score(labels, predictions), "ap_train": average_precision_score(labels, predictions)}


def eval_edge_val_test(model, data, batch_loaders, mode="val"):
    pred_pos, label_pos = evaluate_edge_loader(model, data, batch_loaders[f"{mode}_pos"])
    pred_sns, label_sns = evaluate_edge_loader(model, data, batch_loaders[f"{mode}_neg_sns"])
    pred_mns, label_mns = evaluate_edge_loader(model, data, batch_loaders[f"{mode}_neg_mns"])
    pred_cns, label_cns = evaluate_edge_loader(model, data, batch_loaders[f"{mode}_neg_cns"])

    roc_sns = roc_auc_score(label_pos + label_sns, pred_pos + pred_sns)
    ap_sns = average_precision_score(label_pos + label_sns, pred_pos + pred_sns)
    roc_mns = roc_auc_score(label_pos + label_mns, pred_pos + pred_mns)
    ap_mns = average_precision_score(label_pos + label_mns, pred_pos + pred_mns)
    roc_cns = roc_auc_score(label_pos + label_cns, pred_pos + pred_cns)
    ap_cns = average_precision_score(label_pos + label_cns, pred_pos + pred_cns)

    size = len(pred_pos) // 3
    mixed_labels = label_pos + label_sns[:size] + label_mns[:size] + label_cns[:size]
    mixed_predictions = pred_pos + pred_sns[:size] + pred_mns[:size] + pred_cns[:size]
    roc_mixed = roc_auc_score(mixed_labels, mixed_predictions)
    ap_mixed = average_precision_score(mixed_labels, mixed_predictions)

    return {
        "roc_sns": roc_sns, "ap_sns": ap_sns,
        "roc_mns": roc_mns, "ap_mns": ap_mns,
        "roc_cns": roc_cns, "ap_cns": ap_cns,
        "roc_mixed": roc_mixed, "ap_mixed": ap_mixed,
        "roc_average": (roc_sns + roc_mns + roc_cns + roc_mixed) / 4,
        "ap_average": (ap_sns + ap_mns + ap_cns + ap_mixed) / 4,
    }


def masked_accuracy(logits: Tensor, labels: Tensor):
    if len(logits) == 0:
        return 0.0
    return (logits.argmax(dim=1) == labels).float().mean().item()


def accuracy(logits: Tensor, labels: Tensor, masks):
    return [masked_accuracy(logits[mask], labels[mask]) for mask in masks.values()]


def masked_f1_score(logits: Tensor, labels: Tensor):
    if len(logits) == 0:
        return 0.0
    predictions = logits.argmax(dim=1)
    return f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average="macro", zero_division=0)


def f1_scores(logits: Tensor, labels: Tensor, masks):
    return [masked_f1_score(logits[mask], labels[mask]) for mask in masks.values()]


def masked_auc_roc(auc_score: Tensor, labels: Tensor):
    if len(auc_score) == 0:
        return 0
    return roc_auc_score(labels.cpu().numpy(), auc_score.cpu().numpy())


def auc_rocs(logits: Tensor, labels: Tensor, masks):
    scores = F.softmax(logits, dim=1)[:, 1].detach()
    return [masked_auc_roc(scores[mask], labels[mask]) for mask in masks.values()]
