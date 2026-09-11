from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Config:
    vocab_size: int = 8
    context: int = 16
    width: int = 32
    layers: int = 2
    query_heads: int = 4
    kv_heads: int = 2
    hidden: int = 64
    epsilon: float = 1e-5
    rope_base: float = 10000.0

    def __post_init__(self):
        for name in ("vocab_size", "context", "width", "layers", "query_heads", "kv_heads", "hidden"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % self.query_heads or self.query_heads % self.kv_heads:
            raise ValueError("width must divide into query heads; query heads must group into KV heads")
        if (self.width // self.query_heads) % 2:
            raise ValueError("RoPE requires even head width")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not math.isfinite(self.rope_base) or self.rope_base <= 1:
            raise ValueError("rope_base must be finite and greater than one")
