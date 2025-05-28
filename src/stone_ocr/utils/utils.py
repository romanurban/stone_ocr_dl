import yaml
from collections import Counter
import torch

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_class_weights(dataset, n_classes):
    counts = Counter()
    for _, label in dataset:
        counts[int(label)] += 1

    total = sum(counts.values())
    weights = [0.0] * n_classes
    for i in range(n_classes):
        if counts[i] > 0:
            weights[i] = total / (n_classes * counts[i])
        else:
            weights[i] = 0.0  # or float("inf"), but 0 avoids NaNs
    return torch.tensor(weights, dtype=torch.float32)