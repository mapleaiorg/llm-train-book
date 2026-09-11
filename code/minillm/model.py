"""Readable unfused FP32 CPU path. Attention intentionally materializes T x T."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, epsilon=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.epsilon) * self.weight


def rope(x, base=10000.0):
    """Adjacent pairs; x is [B,H,T,d]. Positions start at zero, no decode cache."""
    d = x.shape[-1]
    frequency = base ** (-torch.arange(0, d, 2, dtype=x.dtype, device=x.device) / d)
    angle = torch.arange(x.shape[-2], dtype=x.dtype, device=x.device)[:, None] * frequency
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * angle.cos() - odd * angle.sin(),
                        even * angle.sin() + odd * angle.cos()), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d = config.width // config.query_heads
        self.q = nn.Linear(config.width, config.width, bias=False)
        self.k = nn.Linear(config.width, config.kv_heads * self.d, bias=False)
        self.v = nn.Linear(config.width, config.kv_heads * self.d, bias=False)
        self.out = nn.Linear(config.width, config.width, bias=False)

    def forward(self, x):
        b, t, _ = x.shape
        c = self.config
        q = rope(self.q(x).reshape(b, t, c.query_heads, self.d).transpose(1, 2), c.rope_base)
        k = rope(self.k(x).reshape(b, t, c.kv_heads, self.d).transpose(1, 2), c.rope_base)
        v = self.v(x).reshape(b, t, c.kv_heads, self.d).transpose(1, 2)
        group = c.query_heads // c.kv_heads
        k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        future = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        probability = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
        result = (probability @ v).transpose(1, 2).contiguous().reshape(b, t, c.width)
        return self.out(result)


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.width, config.hidden, bias=False)
        self.up = nn.Linear(config.width, config.hidden, bias=False)
        self.down = nn.Linear(config.hidden, config.width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.width, config.epsilon)
        self.attention = Attention(config)
        self.mlp_norm = RMSNorm(config.width, config.epsilon)
        self.mlp = SwiGLU(config)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class TinyLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.layers)))
        self.norm = RMSNorm(config.width, config.epsilon)
        # Deliberately untied; the lineage table and checkpoint preserve this choice.
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)

    def forward(self, ids):
        if ids.ndim != 2 or ids.dtype != torch.long or not 0 < ids.shape[1] <= self.config.context or ids.shape[0] == 0:
            raise ValueError("expected nonempty int64 [B,T], T <= context")
        if torch.any(ids < 0) or torch.any(ids >= self.config.vocab_size):
            raise ValueError("token outside vocabulary")
        return self.head(self.norm(self.blocks(self.embedding(ids))))


def cross_entropy(logits, targets):
    """Targets are already shifted by the data layer; do not shift again."""
    if targets.shape != logits.shape[:-1] or targets.dtype != torch.long:
        raise ValueError("targets must be int64 [B,T]")
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())
