from collections import defaultdict
import torch
from utils.metrics import accuracy, edge_evaluation_printer, evaluate_edge, evaluate_hypegraph, f1_scores, hg_evaluation_printer


class Evaluator:
    def __init__(self, args, **kwargs):
        self.args = args
        self.device = args.device

    @torch.no_grad()
    def node_cls_evaluation(self, model, data, masks, result=None):
        if result is None:
            model.eval()
            result = model(data)
        return {"acc": accuracy(result, data.y, masks), "macro_f1": f1_scores(result, data.y, masks)}

    @torch.no_grad()
    def edge_pred_evaluation(self, model, data, batch_loaders, result=None):
        return evaluate_edge(model, data, batch_loaders)

    @torch.no_grad()
    def hg_cls_evaluation(self, model, batch_loaders, result=None):
        return evaluate_hypegraph(model, batch_loaders, self.args)

    def evaluate(self, model, data, seed_split=None, task_type=None, verbose=False):
        if task_type == "node_cls":
            metrics = self.node_cls_evaluation(model, data, seed_split)
            if verbose:
                for name, values in metrics.items():
                    print(f"train_{name}: {100 * values[0]:.2f}%, valid_{name}: {100 * values[1]:.2f}%, test_{name}: {100 * values[2]:.2f}%")
            return defaultdict(list) | metrics

        if task_type == "edge_pred":
            train_metrics, val_metrics, test_metrics = self.edge_pred_evaluation(model, data, seed_split)
            if verbose:
                edge_evaluation_printer(train_metrics, val_metrics, test_metrics)
            return {"train": train_metrics, "val": val_metrics, "test": test_metrics}

        if task_type == "hg_cls":
            result = self.hg_cls_evaluation(model, seed_split)
            if verbose:
                hg_evaluation_printer(result)
            return result

