import hashlib
import json
import torch


class TokenStream:
    """Random windows with a private CPU RNG; every valid start is eligible."""

    def __init__(self, tokens, context, vocab_size, seed=0):
        if type(context) is not int or context <= 0:
            raise ValueError("context must be positive")
        if any(type(t) is not int or not 0 <= t < vocab_size for t in tokens):
            raise ValueError("invalid token ID")
        if len(tokens) <= context:
            raise ValueError("need at least context + 1 tokens")
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context = context
        self.vocab_size = vocab_size
        self.generator = torch.Generator().manual_seed(seed)
        self.batches = 0
        payload = json.dumps([tokens, context, vocab_size], separators=(",", ":"))
        self.identity = hashlib.sha256(payload.encode()).hexdigest()

    def batch(self, batch_size):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        # randint's upper bound is exclusive: last valid start is N - T - 1.
        starts = torch.randint(len(self.tokens) - self.context, (batch_size,), generator=self.generator)
        blocks = torch.stack([self.tokens[i:i + self.context + 1] for i in starts])
        self.batches += 1
        return blocks[:, :-1].contiguous(), blocks[:, 1:].contiguous()

    def state_dict(self):
        return {"identity": self.identity, "rng": self.generator.get_state(), "batches": self.batches}

    def load_state_dict(self, state):
        if state["identity"] != self.identity:
            raise ValueError("dataset/context identity mismatch")
        self.generator.set_state(state["rng"])
        self.batches = state["batches"]
