import math
import torch
from .model import cross_entropy


def optimizer_for(model, learning_rate=0.003):
    # Explicit policy: decay all parameters, including gains; not a universal recipe.
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, foreach=False)


def train_step(model, optimizer, batches):
    """Token-weighted accumulation; checkpoint only after this boundary."""
    batches = list(batches)
    count = sum(y.numel() for _, y in batches)
    if count == 0:
        raise ValueError("at least one target is required")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for x, y in batches:
        loss = cross_entropy(model(x), y)
        weight = y.numel() / count
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss; no optimizer update performed")
        (loss * weight).backward()
        total += loss.item() * weight
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    if not math.isfinite(float(norm)):
        raise FloatingPointError("non-finite gradient")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return total


@torch.no_grad()
def evaluate(model, x, y):
    training = model.training
    try:
        model.eval()
        return cross_entropy(model(x), y).item()
    finally:
        model.train(training)
