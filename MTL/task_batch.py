class TaskBatch:
    def __init__(self, h_prime, task_type, y):
        self.h_prime = h_prime
        self.task_type = task_type
        self.y = y

    def to(self, device):
        self.h_prime = self.h_prime.to(device)
        self.y = self.y.to(device)
        return self
