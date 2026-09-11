"""Run from repo root: PYTHONPATH=code python code/examples/train_tiny.py."""
import argparse
import json
import math
import platform
import torch
from minillm import Config, TinyLM
from minillm.data import TokenStream
from minillm.tokenizer import CharacterTokenizer
from minillm.trainer import optimizer_for, train_step, evaluate
from minillm.checkpoint import save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--checkpoint")
    parser.add_argument("--random-data", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    config = Config()
    tokenizer = CharacterTokenizer()
    stream = TokenStream(tokenizer.encode(tokenizer.alphabet * 64), config.context, config.vocab_size, 43)
    model = TinyLM(config)
    optimizer = optimizer_for(model)
    train_rng = torch.Generator().manual_seed(44)
    eval_rng = torch.Generator().manual_seed(45)

    def fresh(generator, batch):
        block = torch.randint(config.vocab_size, (batch, config.context + 1), generator=generator)
        return block[:, :-1], block[:, 1:]

    heldout = fresh(eval_rng, 256) if args.random_data else stream.batch(32)
    before = evaluate(model, *heldout)
    for _ in range(args.steps):
        batch = fresh(train_rng, 8) if args.random_data else stream.batch(8)
        train_step(model, optimizer, [batch])
    after = evaluate(model, *heldout)
    if args.checkpoint:
        if args.random_data:
            parser.error("random-data sanity uses a separate RNG; checkpoint export is disabled")
        save(args.checkpoint, model, optimizer, stream, tokenizer, args.steps)
    print(json.dumps({"evidence": "MEASURED", "device": "cpu", "dtype": "float32",
                      "python": platform.python_version(), "torch": str(torch.__version__),
                      "seed": 42, "steps": args.steps, "random_data": args.random_data,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "evaluation_before": before, "evaluation_after": after,
                      "uniform_entropy_derived": math.log(config.vocab_size)}, indent=2))


if __name__ == "__main__":
    main()
