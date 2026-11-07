import time
import torch
import torch.nn.functional as F


class MultiTaskExpAgent:
    def __init__(self, args, task_weights=None):
        self.args = args
        self.device = torch.device(args.device)
        self.task_weights = task_weights or {}
        self.history = []
        self.train_time = 0.0

    def train_epoch(self, model, task_loaders, optimizer):
        model.train()
        tasks = list(task_loaders)
        batch_counts = [len(task_loaders[task]) for task in tasks]
        if len(set(batch_counts)) != 1:
            raise ValueError(f"Task loaders have different batch counts: {dict(zip(tasks, batch_counts))}")

        weights = {task: float(self.task_weights.get(task, 1.0)) for task in tasks}
        weight_sum = sum(weights.values())
        loss_sums = torch.zeros(len(tasks), device=self.device)

        for batches in zip(*(task_loaders[task] for task in tasks)):
            optimizer.zero_grad(set_to_none=True)

            for index, (task, batch) in enumerate(zip(tasks, batches)):
                batch = batch.to(self.device)
                loss = F.cross_entropy(model(batch), batch.y.view(-1).long())
                weighted_loss = weights[task] * loss / weight_sum
                weighted_loss.backward()
                loss_sums[index] += loss.detach()

            optimizer.step()

        steps = batch_counts[0]
        mean_losses = (loss_sums / steps).cpu().tolist()
        task_losses = dict(zip(tasks, mean_losses))
        total_loss = sum(weights[task] * task_losses[task] for task in tasks) / weight_sum
        return total_loss, task_losses

    def build_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=getattr(self.args, "wd", 0.0))

    def training(self, model, task_loaders, reset_parameters=True):
        model = model.to(self.device)
        if reset_parameters:
            model.reset_parameters()
        optimizer = self.build_optimizer(model)

        self.history = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            total_loss, task_losses = self.train_epoch(model, task_loaders, optimizer)
            self.history.append({"epoch": epoch, "loss": total_loss, "task_losses": task_losses})
            if epoch % self.args.display_step == 0 or epoch == self.args.epochs:
                details = " | ".join(f"{task}: {loss:.4f}" for task, loss in task_losses.items())
                print(f"Epoch {epoch:03d} | Loss: {total_loss:.4f} | {details}")

        self.train_time = time.time() - start_time
        print(f"Training Time: {self.train_time:.2f}s")
        return model
