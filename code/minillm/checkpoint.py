"""Same-version, same-CPU, optimizer-boundary resume; trusted local files only."""
from dataclasses import asdict
import os
from pathlib import Path
import tempfile
import torch


def validate_contract(model, stream, tokenizer):
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("resume contract is CPU FP32 only")
    if len(tokenizer.alphabet) != model.config.vocab_size:
        raise ValueError("tokenizer vocabulary mismatch")
    if stream.context != model.config.context or stream.vocab_size != model.config.vocab_size:
        raise ValueError("data/model shape contract mismatch")


def save(path, model, optimizer, stream, tokenizer, step):
    validate_contract(model, stream, tokenizer)
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("checkpoint must follow a completed optimizer step with cleared gradients")
    payload = {"schema": 1, "torch_version": str(torch.__version__), "step": step,
               "config": asdict(model.config), "tokenizer": tokenizer.identity,
               "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "optimizer_type": type(optimizer).__name__, "stream": stream.state_dict(),
               "rng": torch.get_rng_state()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            torch.save(payload, output)
        # Atomic visibility on the same filesystem; no power-loss durability claim.
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load(path, model, optimizer, stream, tokenizer):
    validate_contract(model, stream, tokenizer)
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("restore requires a clean optimizer boundary")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["schema"] != 1 or payload["torch_version"] != str(torch.__version__):
        raise ValueError("checkpoint schema/runtime mismatch")
    if payload["config"] != asdict(model.config) or payload["tokenizer"] != tokenizer.identity:
        raise ValueError("model/tokenizer identity mismatch")
    if payload["stream"]["identity"] != stream.identity:
        raise ValueError("dataset/context identity mismatch")
    if payload["optimizer_type"] != type(optimizer).__name__:
        raise ValueError("optimizer type mismatch")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    stream.load_state_dict(payload["stream"])
    # Restore last: constructing the model may have consumed global RNG values.
    torch.set_rng_state(payload["rng"])
    return payload["step"]
